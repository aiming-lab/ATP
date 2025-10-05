import re

class DummyRewardFunction:
    def __init__(self, config, **kwargs):
        self.config = config
        self.__name__ = "DummyRewardFunction"
        
    def __call__(self, completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]
    
class ThinkRewardFunction:
    def __init__(self, config, **kwargs):
        self.config = config
        self.__name__ = "ThinkRewardFunction"
        
    def __call__(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            if isinstance(completion[0]["content"], str):
                answer = completion[0]["content"]
            else:
                answer = completion[0]["content"]["text"]
            if re.search(r"<think>.+?</think>", answer, re.DOTALL) is not None:
                reward = 1
            else:
                reward = 0
            rewards.append(reward)
            
        return rewards