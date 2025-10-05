import os
import sys
import json
import datetime
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.extract_json import extract_choice_from_response
from utils.config import name_to_class

def save_results(cfg, datetime_str, data, final=False):
    if final:
        filename = f'{cfg.result_dir}/{datetime_str}.json'
    else:
        os.makedirs(cfg.temp_result_dir, exist_ok=True)
        filename = f'{cfg.temp_result_dir}/{datetime_str}.json'

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    if final:
        print(f"[INFO] Raw results saved to {cfg.result_dir}/{datetime_str}.json")

def run_worker(rank, gpu_id, cfg_dict, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    
    cfg = OmegaConf.create(cfg_dict)
    model = name_to_class(cfg.model)
    dataset = name_to_class(cfg.dataset)
    
    print(f"[Worker {rank}] Deployed model on GPU {gpu_id}, ready for inference.")

    while True:
        task = task_queue.get()
        if task is None:
            break
        
        idx, d = task
        choices_list = dataset.get_choices(d)
        
        result = { "supervised": None, "better": None }
        messages = dataset.get_messages(d)
        for j in range(cfg.max_retries):
            try:
                messages = dataset.get_supervised_prompt(d)
                answer, _ = model.predict_message(messages=messages)
                choice = extract_choice_from_response(answer, choices=choices_list)
                if choice == d["supervised_choice"].lower():
                    result["supervised"] = answer
                    break
            except Exception as e:
                print(f"[Worker {rank} on GPU {gpu_id}] Retry {j+1} for player {idx}: {e}")
                
        for j in range(cfg.max_retries):
            try:
                messages = dataset.get_better_prompt(d)
                answer, _ = model.predict_message(messages=messages)
                choice = extract_choice_from_response(answer, choices=choices_list)
                if choice == d["better_choice"].lower():
                    result["better"] = answer
                    break
            except Exception as e:
                print(f"[Worker {rank} on GPU {gpu_id}] Retry {j+1} for player {idx}: {e}")
        result_queue.put((idx, result))

@hydra.main(config_path="../config", config_name="dpo_sa_gen", version_base="1.2")
def main(cfg):
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(cfg.result_dir, exist_ok=True)

    dataset_cls = name_to_class(cfg.dataset)
    full_data = dataset_cls.load_dataset()
    
    gpu_list = list(map(int, cfg.cuda_visible_devices.split(",")))
    num_workers = min(len(gpu_list), cfg.max_workers)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for idx, d in enumerate(full_data):
        task_queue.put((idx, d))
    for _ in range(num_workers):
        task_queue.put(None)
        
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    processes = []
    for i in range(num_workers):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = mp.Process(target=run_worker, args=(i, gpu_id, cfg_dict, task_queue, result_queue))
        p.start()
        processes.append(p)
        print(f"[Main] Starting worker {i} on GPU {gpu_id}...")

    results = [{} for _ in full_data]
    pbar = tqdm(total=len(full_data), desc='Processing Games', unit='item')
    for _ in range(len(full_data)):
        idx, result = result_queue.get()
        results[idx] = result
        pbar.update(1)
    pbar.close()

    for idx, result in enumerate(results):
        if result:
            full_data[idx]["result"] = result

    for p in processes:
        p.join()

    save_results(cfg, datetime_str, full_data, final=True)

if __name__ == "__main__":
    mp.set_start_method("fork", force=True) 
    main()
