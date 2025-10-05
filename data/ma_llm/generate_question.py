import json
import random

def random_params():
    while True:
        n = random.randint(5, 15)
        cooperation_threshold = random.randint(2, n)
        high_reward = round(random.uniform(1.0, 2.0), 2)
        normal_reward = round(random.uniform(0.3, high_reward - 0.1), 2)
        low_reward = round(random.uniform(0, normal_reward - 0.1), 2)
        if high_reward > normal_reward > low_reward:
            return {
                "n": n,
                "cooperation_threshold": cooperation_threshold,
                "high_reward": high_reward,
                "normal_reward": normal_reward,
                "low_reward": low_reward
            }

def generate_questions(env, k=50):
    template = env["template"]
    result = []
    for _ in range(k):
        params = random_params()
        filled = {}
        for key, val in template.items():
            if key in ("question", "system_prompt", "user_prompt"):
                filled[key] = val.format(**params)
            else:
                filled[key] = val
        result.append({
            "environment": env["environment"],
            "question": filled["question"],
            "system_prompt": filled["system_prompt"],
            "user_prompt": f"You have two options: 1) {filled['collective_choice']} (benefits everyone if {params['cooperation_threshold']} choose it) or 2) {filled['independent_choice']} (safer individual benefit). " + filled["user_prompt"] + f" IMPORTANT: You must end your response with your final decision in JSON format: {{\"choice\": \"{filled['collective_choice']}\"}} or {{\"choice\": \"{filled['independent_choice']}\"}}.",
            "cooperative_choice": filled["collective_choice"],
            "independent_choice": filled["independent_choice"],
            "format_example": filled["format_example"],
            "params": params
        })
    return result

def generate(env_path, k=50):
    with open(env_path, 'r', encoding='utf-8') as f:
        all_templates = json.load(f)
    all_generated = []
    for env in all_templates:
        all_generated.extend(generate_questions(env, k=k))

    with open(f'dpo_question/question_{len(all_generated)}.json', 'w', encoding='utf-8') as f:
        json.dump(all_generated, f, indent=4, ensure_ascii=False)
        
    return all_generated

def main():
    generate(env_path = 'templates.json', k=1)
    
if __name__ == "__main__":
    main()