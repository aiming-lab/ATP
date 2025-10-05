from trl import GRPOTrainer, GRPOConfig
import inspect

def filter_config_for_class(config, target_class):
    sig = inspect.signature(target_class.__init__)
    valid_params = [p for p in sig.parameters.keys() if p != "self"]
    filtered_kwargs = {k: v for k, v in config.items() if k in valid_params}
    return filtered_kwargs

class GRPOTrain:
    def __init__(self, config):
        self.config = config
        grpo_kwargs = filter_config_for_class(self.config, GRPOConfig)
        self.training_args = GRPOConfig(**grpo_kwargs)
        
    def train(self, model, dataset, reward_func):
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=self.training_args,
            train_dataset=dataset,
        )
        trainer.train()