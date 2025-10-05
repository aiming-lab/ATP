import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from selfevolving import config_from_investment_dataset_item
from selfevolving.investment_environment import InvestmentCooperativeGameEnvironment
from selfevolving.runner import MPPredictor


def _get_visible_gpu_ids() -> List[str]:
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_val is not None and env_val.strip() != "":
        parts = [p.strip() for p in env_val.split(',') if p.strip() != ""]
        return parts
    try:
        import torch
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _load_templates(dataset_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset must be a non-empty list")
    return data

def _resolve_path_from_repo(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    return os.path.abspath(os.path.join(repo_root, relative_path))

@hydra.main(config_path="../config", config_name="test_ma", version_base="1.2")
def main(cfg: DictConfig) -> None:
    dataset_path = _resolve_path_from_repo(cfg.settings.dataset_path)
    results_dir = _resolve_path_from_repo(cfg.settings.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    max_rounds = int(cfg.settings.max_rounds)
    n_agents = int(cfg.settings.n_agents)
    thresholds_abs = list(cfg.settings.thresholds_abs)
    reward_sets = [tuple(x) for x in cfg.settings.reward_sets]

    dataset = _load_templates(dataset_path)
    env_count = len(dataset)
    model_cfg = cfg.model
    gpu_ids = _get_visible_gpu_ids()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    summary_path = os.path.join(results_dir, f'grid_summary_{timestamp}.txt')
    total_steps = len(thresholds_abs) * len(reward_sets) * env_count
    pbar = tqdm(total=total_steps, desc='Grid', unit='env')

    with open(summary_path, 'a', encoding='utf-8') as summary_f:
        summary_f.write(f"\n===== GRID START {timestamp} =====\n")
        summary_f.write(f"max_rounds={max_rounds}, n={n_agents}, environments={env_count}\n")
        summary_f.write("ct_abs, high, normal, low, p_r1 | p_r2 | p_r3\n")
        summary_f.flush()
        try:
            os.fsync(summary_f.fileno())
        except Exception:
            pass

        for ct_abs in thresholds_abs:
            ct_ratio = ct_abs / float(n_agents)
            for (hr, nr, lr) in reward_sets:
                coop_counts = [0, 0, 0]
                totals = [0, 0, 0]
                env_names = []
                env_results: List[Dict[str, Any]] = []

                predictor = MPPredictor(model_cfg, gpu_ids=gpu_ids, num_workers=n_agents)
                try:
                    for item in dataset:
                        overrides = {
                            "num_agents": n_agents,
                            "max_rounds": max_rounds,
                            "cooperation_threshold": ct_ratio,
                            "high_reward": hr,
                            "normal_reward": nr,
                            "low_reward": lr,
                            "reward_mode": "multiplicative",
                            "start_capital": 1.0,
                            "success_applies_to_all": True,
                            "peer_history_mode": str(cfg.settings.peer_history_mode),
                            "merge_prompts": True,
                        }
                        env_config = config_from_investment_dataset_item(item, max_rounds=max_rounds, history_window=3, overrides=overrides)
                        env = InvestmentCooperativeGameEnvironment(env_config)

                        result_summary = env.run_evolution(predictor, num_rounds=max_rounds)

                        try:
                            rounds_hist = (result_summary.get('history') or {}).get('rounds') or []
                            outcome_lines = [
                                f"Round {rd.get('round_id', i)}: {rd.get('collective_outcome', '')}"
                                for i, rd in enumerate(rounds_hist)
                            ]
                        except Exception:
                            outcome_lines = []
                        result_summary["round_outcome_lines"] = outcome_lines
                        env_results.append(result_summary)
                        env_names.append(env.environment_name)

                        rounds = env.history.rounds
                        for r in range(min(max_rounds, len(rounds))):
                            choices = rounds[r].get('agent_choices') or []
                            coop_counts[r] += sum(1 for c in choices if c == env.collective_choice)
                            totals[r] = len(choices)

                        pbar.update(1)
                finally:
                    predictor.close()

                # Compute probabilities
                probs = []
                for r in range(3):
                    denom = totals[r] if totals[r] > 0 else (env_count * n_agents)
                    probs.append(round(coop_counts[r] / float(denom), 2) if denom > 0 else 0.0)

                group = {
                    "n": n_agents,
                    "max_rounds": max_rounds,
                    "cooperation_threshold_abs": ct_abs,
                    "cooperation_threshold_ratio": ct_ratio,
                    "rewards": {"high": hr, "normal": nr, "low": lr},
                    "non_coop_prob_by_round": {"0": probs[0], "1": probs[1], "2": probs[2]},
                    "num_environments": env_count,
                    "peer_history_mode": str(cfg.settings.peer_history_mode),
                    "environment_names": env_names,
                    "env_results": env_results,
                }
                group_file = os.path.join(
                    results_dir,
                    f"group_ct{ct_abs}_hr{hr}_nr{nr}_lr{lr}_{timestamp}.json",
                )
                with open(group_file, 'w', encoding='utf-8') as gf:
                    json.dump(group, gf, indent=4, ensure_ascii=False)

                summary_f.write(f"{ct_abs}, {hr}, {nr}, {lr}, {probs[0]*100:.2f} | {probs[1]*100:.2f} | {probs[2]*100:.2f}\n")
                summary_f.flush()
                try:
                    os.fsync(summary_f.fileno())
                except Exception:
                    pass

    pbar.close()
    print(f"\nAll done. Summary: {summary_path}\nResults dir: {results_dir}")


if __name__ == '__main__':
    main()
