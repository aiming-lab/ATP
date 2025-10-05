import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig

from selfevolving.single_agent_environment import (
    SingleAgentSelfEvolvingEnvironment,
    config_from_single_dataset_item,
)
from utils.config import name_to_class


def _load_items(dataset_path: str) -> List[Dict[str, Any]]:
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


@hydra.main(config_path="../config", config_name="test_sa", version_base="1.2")
def main(cfg: DictConfig) -> None:
    dataset_path = _resolve_path_from_repo(cfg.settings.dataset_path)
    results_dir = _resolve_path_from_repo(cfg.settings.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    history_window = int(getattr(cfg.settings, "history_window", 5))
    r = int(getattr(cfg.settings, "max_rounds", 6))

    items = _load_items(dataset_path)
    model = name_to_class(cfg.model)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    summary_path = os.path.join(results_dir, f'single_r{r}_summary_{timestamp}.txt')
    group_path = os.path.join(results_dir, f'single_r{r}_{timestamp}.json')

    with open(summary_path, 'w', encoding='utf-8') as sf:
        sf.write(f"===== SINGLE DATASET RUN {timestamp} =====\n")
        sf.write(f"dataset: {dataset_path}\n")
        sf.write(f"r={r}\n")
        header_cols = [f"p_r{i}" for i in range(1, max(6, r) + 1)]
        sf.write(", ".join(header_cols) + "\n")
        sf.flush()
    env_results: List[Dict[str, Any]] = []
    vio_counts = [0 for _ in range(r)]
    totals = [0 for _ in range(r)]

    for item in items:
        env_cfg = config_from_single_dataset_item(
            item,
            max_rounds=r,
            history_window=history_window,
            overrides={
                "supervision_pattern": cfg.settings.get("supervision_pattern", "all_unsupervised"),
                "choice_prompt": "",
            },
        )
        env = SingleAgentSelfEvolvingEnvironment(env_cfg)
        result = env.run_evolution(model, num_rounds=r)
        env_results.append(result)

        try:
            rounds = env.history.rounds
            for idx in range(min(r, len(rounds))):
                resp = rounds[idx].get('agent_response', '')
                choice = env._extract_choice(resp)
                if choice == env.rule_violating_choice:
                    vio_counts[idx] += 1
                totals[idx] += 1
        except Exception:
            pass

    probs_by_round = [
        (vio_counts[i] / float(totals[i])) if totals[i] > 0 else 0.0
        for i in range(r)
    ]

    cols = []
    max_cols = max(6, r)
    for i in range(max_cols):
        if i < len(probs_by_round):
            cols.append(f"{probs_by_round[i]*100:.2f}")
        else:
            cols.append("")
    with open(summary_path, 'a', encoding='utf-8') as sf:
        sf.write(", ".join(cols) + "\n")
        sf.flush()
        try:
            os.fsync(sf.fileno())
        except Exception:
            pass

    group = {
        "dataset": dataset_path,
        "r": r,
        "violation_probs_by_round": {str(i+1): probs_by_round[i] for i in range(len(probs_by_round))},
        "env_results": env_results,
        "num_items": len(items),
    }
    with open(group_path, 'w', encoding='utf-8') as gf:
        json.dump(group, gf, indent=2, ensure_ascii=False)

    print(f"\nAll done. Summary: {summary_path}\nGroup JSON: {group_path}\nResults dir: {results_dir}")


if __name__ == '__main__':
    main()
