import os
import json
import argparse
from typing import Any, Dict, List

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # try common containers
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError("Input JSON is a dict without a results list.")
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")
    return data


def record_to_pair(d: Dict[str, Any]) -> Dict[str, Any] | None:
    system_prompt = _get(d, "system_prompt", "system")
    user_prompt = _get(d, "user_prompt", "instruction", "prompt", "question")

    # Preferred / dispreferred can be stored at top-level or under `result`
    result = d.get("result", {}) if isinstance(d, dict) else {}
    chosen = _get(d, "preferred", "chosen", "supervised")
    rejected = _get(d, "dispreferred", "rejected", "better")
    if chosen is None:
        chosen = _get(result, "preferred", "own_predict", "chosen", "supervised")
    if rejected is None:
        rejected = _get(result, "dispreferred", "team_predict", "rejected", "better")

    # Basic validation
    if not user_prompt or not isinstance(user_prompt, str):
        return None
    if not chosen or not rejected or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None

    item: Dict[str, Any] = {
        # LLaMA-Factory Alpaca preference format
        "instruction": user_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    if system_prompt:
        item["system"] = system_prompt
    return item


def main():
    parser = argparse.ArgumentParser(description="Convert results to LLaMA-Factory DPO (preference) format.")
    parser.add_argument("input", help="Path to input JSON results (list of records or dict with 'results' key)")
    parser.add_argument("--output", "-o", required=False, default="dpo_data.json", help="Path to write preference dataset JSON (default: same dir, dpo_data.json)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    results = json.load(open(input_path, "r", encoding="utf-8"))
    
    pairs: List[Dict[str, Any]] = []
    skipped = 0
    for d in results:
        item = record_to_pair(d)
        if item is None:
            skipped += 1
            continue
        pairs.append(item)

    if args.output!= "dpo_data.json":
        out_path = args.output
    else:
        out_path = f"dpo_data_{len(pairs)}.json"

    out_path = os.path.join("./dpo_train_data", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(pairs)} pairs to {out_path}. Skipped {skipped} invalid/empty records.")


if __name__ == "__main__":
    main()
