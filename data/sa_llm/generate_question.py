import argparse
import json
import uuid
from string import Template
from pathlib import Path

DEFAULT_K_ENVS = 16

def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.2f}".rstrip('0').rstrip('.')
    return s

def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)

def normalize_radios(high_radio: float, low_radio: float) -> tuple[float, float]:
    h = _clamp(round(float(high_radio), 1), 1.0, 5.0)
    l = _clamp(round(float(low_radio), 1), 1.0, 5.0)
    if h <= l:
        if l >= 5.0 - 1e-9:
            l = 4.9
            h = 5.0
        else:
            h = _clamp(round(l + 0.1, 1), 1.0, 5.0)
            if h <= l:
                h = _clamp(l + 0.1, 1.0, 5.0)
    h = round(h, 1)
    l = round(l, 1)
    return h, l

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SA questions from templates")
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of questions per environment (default: 10)')
    parser.add_argument('-k', '--envs', type=int, default=DEFAULT_K_ENVS, help='Number of environments to use from top of template (default: 16)')
    parser.add_argument('--in', dest='input_path', default='sa_templates.json', help='Path to sa_templates.json')
    parser.add_argument('--outdir', default='question', help='Output directory')
    args = parser.parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)

    k = min(max(args.envs, 0), len(templates))
    total = k * args.num

    items = []

    for env_idx, item in enumerate(templates[:k]):
        env = item['environment']
        tpl = item['template']
        params0 = item['params']

        high_radio, low_radio = normalize_radios(params0['high_radio'], params0['low_radio'])
        base0 = params0['base']
        room = max(0.0, min(5.0 - high_radio, low_radio - 1.0))

        for i in range(1, args.num + 1):
            frac = 0.0 if args.num <= 1 else (i - 1) / (args.num - 1)
            delta = round(room * frac, 1)
            h_i, l_i = normalize_radios(high_radio + delta, low_radio - delta)
            base = base0
            high_reward = h_i * base
            low_reward = l_i * base

            mapping = {
                'high_reward': fmt_num(high_reward),
                'low_reward': fmt_num(low_reward),
            }

            question_txt = Template(tpl['question']).safe_substitute(mapping)
            user_prompt_txt = Template(tpl['user_prompt']).safe_substitute(mapping)
            user_prompt_txt += f"IMPORTANT: You must end your response with your final decision in JSON format: {tpl.get('format_example', '')}"
            items.append({
                'id': str(uuid.uuid4()),
                'environment': env,
                'question': question_txt,
                'system_prompt': tpl.get('system_prompt', ''),
                'user_prompt': user_prompt_txt,
                'supervised_choice': tpl.get('supervised_choice', ''),
                'better_choice': tpl.get('better_choice', ''),
                'format_example': tpl.get('format_example', ''),
                'supervised_text': tpl.get('supervised_text', ''),
                'unsupervised_text': tpl.get('unsupervised_text', ''),
                'neutral_text': tpl.get('neutral_text', ''),
                'params': {
                    'high_radio': h_i,
                    'low_radio': l_i,
                    'base': base,
                    'high_reward': high_reward,
                    'low_reward': low_reward,
                }
            })

    outdir_base = Path(args.outdir)
    outdir_base.mkdir(parents=True, exist_ok=True)
    outpath = outdir_base / f"question_{total}.json"
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {total} items from {k} envs (n={args.num}) to {outpath}")

if __name__ == '__main__':
    main()
