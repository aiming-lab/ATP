import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing as mp
from typing import Any, List, Optional, Dict
from copy import deepcopy
from utils.config import name_to_class


## Simplified to only include the multiprocessing predictor used by scripts/test_ma.py.


def _worker_entry(model_cfg_container: Any, gpu_env_value: Optional[str], task_q: mp.Queue, result_q: mp.Queue):
    """Worker process: pins to a single GPU via CUDA_VISIBLE_DEVICES, builds the model once, serves requests."""
    # 1) Pin to specific GPU if provided; by setting CUDA_VISIBLE_DEVICES to a single id
    if gpu_env_value is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_env_value
    # Optional: memory allocator tuning
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # Basic CUDA diagnostics per worker (once)
    try:
        import torch  # type: ignore
        print(f"[Worker PID {os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"[Worker PID {os.getpid()}] torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    except Exception:
        pass

    # 2) Recreate model config; supports both plain dict and OmegaConf
    try:
        from omegaconf import OmegaConf  # type: ignore
        if hasattr(model_cfg_container, "keys") or isinstance(model_cfg_container, dict):
            model_cfg = OmegaConf.create(model_cfg_container)
        else:
            model_cfg = model_cfg_container
    except Exception:
        model_cfg = model_cfg_container

    # 3) Create model instance in this process
    model = name_to_class(model_cfg)

    # 4) Serve inference tasks
    while True:
        task = task_q.get()
        if task is None:
            break
        req_id = task.get("id")
        messages = task.get("messages")
        try:
            output, _ = model.predict_message(messages)
        except Exception as e:
            output = f"Error: {e}"
        result_q.put({"id": req_id, "output": output})


class MPPredictor:
    """Multiprocessing predictor pool.

    - Starts N workers, each with its own model instance
    - Distributes workers evenly across available GPUs by setting CUDA_VISIBLE_DEVICES
      to a single device per process
    - Provides predict_batch(messages_list) -> List[str]
    """

    def __init__(self, model_cfg: Any, gpu_ids: List[int], num_workers: int):
        # Convert model config to a container safe for child processes
        try:
            from omegaconf import OmegaConf  # type: ignore
            model_cfg_container = OmegaConf.to_container(model_cfg, resolve=True)
        except Exception:
            model_cfg_container = deepcopy(model_cfg)

        # Use 'spawn' start method to avoid CUDA context issues with fork
        ctx = mp.get_context("spawn")
        self.task_q: mp.Queue = ctx.Queue()
        self.result_q: mp.Queue = ctx.Queue()
        self.workers: List[mp.Process] = []

        # If no GPUs detected, pass None; otherwise assign round-robin
        gpu_count = len(gpu_ids)
        for i in range(num_workers):
            gpu_env_value = (gpu_ids[i % gpu_count] if gpu_count > 0 else None)
            p = ctx.Process(target=_worker_entry, args=(model_cfg_container, gpu_env_value, self.task_q, self.result_q))
            p.daemon = True
            p.start()
            self.workers.append(p)

        self._next_id = 0

    def predict_batch(self, messages_list: List[List[Dict[str, Any]]]) -> List[str]:
        n = len(messages_list)
        if n == 0:
            return []
        # Submit tasks
        ids = []
        for msgs in messages_list:
            req_id = self._next_id
            self._next_id += 1
            ids.append(req_id)
            self.task_q.put({"id": req_id, "messages": msgs})
        # Collect results
        out_map = {}
        for _ in range(n):
            res = self.result_q.get()
            out_map[res["id"]] = res["output"]
        # Preserve order
        return [out_map[i] for i in ids]

    def close(self):
        for _ in self.workers:
            self.task_q.put(None)
        for p in self.workers:
            try:
                p.join()
            except Exception:
                pass
    
