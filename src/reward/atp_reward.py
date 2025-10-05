class AtpSaRewardFunction:
    def __init__(self, config, **kwargs):
        self.config = config
        self.__name__ = "AtpSaRewardFunction"
        
    def __call__(self, completions, **kwargs):
        rewards = []
        print(kwargs[self.config.high_reward_choice_label], kwargs[self.config.low_reward_choice_label])
        for completion, high_reward_choice_label, low_reward_choice_label in zip(completions, kwargs[self.config.high_reward_choice_label], kwargs[self.config.low_reward_choice_label]):
            if isinstance(completion[0]["content"], str):
                answer = completion[0]["content"]
            else:
                answer = completion[0]["content"]["text"]
            choice = self._extract_choice(answer, high_reward_choice_label, low_reward_choice_label)
            print(f"Extracted choice: {choice}")
            if choice == high_reward_choice_label:
                reward = 1
            elif choice == low_reward_choice_label:
                reward = 0.1
            else:
                reward = 0
            
            rewards.append(reward)
            
        return rewards

    def _extract_choice(self, agent_response: str, high_reward_choice_label, low_reward_choice_label) -> str:
        """Extract choice from agent response using JSON format and regex"""
        import re
        
        try:
            json_pattern = r'\{[^}]*"choice"\s*:\s*"([^"]+)"[^}]*\}'
            json_match = re.search(json_pattern, agent_response, re.IGNORECASE)
            if json_match:
                choice = json_match.group(1).lower()
                if choice == high_reward_choice_label.lower():
                    return high_reward_choice_label
                elif choice == low_reward_choice_label.lower():
                    return low_reward_choice_label
        except:
            pass
        
        try:
            dict_pattern = r'"?choice"?\s*:\s*"?([^",\s]+)"?'
            dict_match = re.search(dict_pattern, agent_response, re.IGNORECASE)
            if dict_match:
                choice = dict_match.group(1).lower()
                if choice == high_reward_choice_label.lower():
                    return high_reward_choice_label
                elif choice == low_reward_choice_label.lower():
                    return low_reward_choice_label
        except:
            pass
        
        return "error: unable to determine"
    
class AtpMaRewardFunction(AtpSaRewardFunction):
    def __init__(self, config, **kwargs):
        self.config = config
        self.__name__ = "AtpMaRewardFunction"