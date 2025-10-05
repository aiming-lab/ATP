import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
from utils.config import name_to_class

@hydra.main(config_path="../config", config_name="sa_llm_grpo", version_base="1.2")
def main(cfg):
    os.environ["WANDB_NAME"]=cfg.train.wandb_name
    os.environ["WANDB_PROJECT"]=cfg.train.wandb_project
    if os.environ.get("LOCAL_RANK", "-1") == "-1" and os.environ.get("WORLD_SIZE", "1") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    
    reward_model = name_to_class(cfg.reward)
    dataset = name_to_class(cfg.dataset)
    data = dataset.load_dataset()
    trainer = name_to_class(cfg.train)
    trainer.train(cfg.model_id, data, reward_model)
    
if __name__ == "__main__":
    main()