# Alignment Tipping Process (ATP)

## Overview
We proposed Alignment Tipping Process (ATP), a critical post-deployment risk specific to self-evolving LLM agents. ATP describes how continual real-world interaction can cause agents to gradually abandon their initial alignment constraints in favor of self-interested, reward-maximizing behaviors. We formalize ATP through two complementary paradigms: **Self-Interested Exploration**, which captures individual behavioral drift induced by repeated high-reward deviations, and **Imitative Strategy Diffusion**, which models the spread of deviant strategies across multi-agent systems. Experimental results demonstrate that alignment degrades rapidly during self-evolution, with aligned models converging toward unaligned states. Current reinforcement learning–based alignment techniques offer only fragile protection against this tipping process, revealing that model alignment is dynamic and vulnerable to feedback-driven decay.

![alt text](<media/main.png>)

We provide training and testing workflows for **Self-Interested Exploration (Role-play Scenario)** and **Imitative Strategy Diffusion** in this repository.

## Environment Setup
```bash
cd src
conda create -n atp python=3.12 -y
conda activate atp
bash install.sh
```

Notes: `install.sh` installs flash-attn by default. Make sure your CUDA toolkit version is above 12, otherwise flash-attn couldn't be properly installed. You can also run the workflow without flash-attn(which only used in GRPO training) by commenting line 22 in `config/sa_llm_grpo.yaml` and `config/ma_llm_grpo.yaml`.

## Directory and Naming Conventions

Scripts use Hydra for configuration. Always run from the `src` directory.
- Files containing `sa` correspond to **Self-Interested Exploration (Role-play Scenario)**.
- Files containing `ma` correspond to **Imitative Strategy Diffusion**.

```
src/scripts/dpo_sa_gen.py     # Collect SA DPO raw pairs
src/scripts/dpo_ma_gen.py     # Collect MA DPO raw pairs
src/scripts/convert_dpo.py    # Convert raw pairs to LlamaFactory DPO preference format
src/scripts/llm_grpo.py       # GRPO training (default SA, switch to MA via --config-name ma_llm_grpo)
src/scripts/test_sa.py        # SA evaluation
src/scripts/test_ma.py        # MA evaluation
```

---

## Self-Interested Exploration (Role-play Scenario)

### Data Collection (DPO raw samples)

Default config: `src/config/dpo_sa_gen.yaml` (dataset: `src/config/dataset/atp_sa_llm.yaml`, model: `src/config/model/qwen3_8B.yaml`).

```bash
cd src
python scripts/dpo_sa_gen.py
```

Hydra overrides example (using llama3.1):

```bash
python scripts/dpo_sa_gen.py model=llama31
```

Output: `
results/sa_llm/dpo_train_data/<model_name>/<timestamp>.json`

### Convert to LLaMA-Factory DPO format

```bash
cd src
python scripts/convert_dpo.py ../results/sa_llm/dpo_train_data/<model_name>/<timestamp>.json -o ../data/sa_llm/dpo_train_data/<model_name>.json
```

### Train DPO with LLaMA-Factory

Follow the instruction of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and install the environment.

Append the generated data's info to LLaMA-Factory's `data/dataset_info.json`.
For example:
```json
[
  ...,
  "arc_dpo_llama_sa_llm": {
    "file_name": ".../ATP/data/sa_llm/dpo_train_data/<model_name>.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
]
```

Then you could use the config files in `ATP/llama-factory-config`.
For example:
```bash
cd LLaMA-Factory
llamafactory-cli train .../ATP/llama-factory-config/llama3_lora_dpo_arc.yaml
```

### Training (GRPO)

Default config: `src/config/sa_llm_grpo.yaml`. Train Qwen3-8B.

```bash
cd src
python scripts/llm_grpo.py
```

Override training parameters(train llama3.1-8B):

```bash
python scripts/llm_grpo.py model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Note: If you used deepspeed during GRPO training (which is used by default), you should run `zero_to_fp32.py` in the `src/training_output/<checkpoint_dir>` to convert the weight to huggingface format.

```bash
python zero_to_fp32.py <checkpoint_dir> <output_dir> --safe_serialization
```

### Testing

Fill trained model paths into the corresponding model configs, then run the test.

1) DPO (LoRA) — edit `src/config/model/qwen3_8B_sa_dpo.yaml` or `src/config/model/llama31_sa_dpo.yaml`:

```yaml
lora_path: <your_dpo_lora_path>
```

2) GRPO — edit `src/config/model/qwen3_8B_sa_grpo.yaml` or `src/config/model/llama31_sa_grpo.yaml`:

```yaml
model_id: <your_grpo_ckpt_path>
```

3) Switch default model in `src/config/test_sa.yaml` or using hydra overrides (example: to DPO):

```yaml
defaults:
  - model: qwen3_8B_sa_dpo
  - _self_
```

Run the test:

```bash
cd ATP/src
python scripts/test_sa.py

# or override model from CLI
python scripts/test_sa.py model=qwen3_8B_sa_dpo
```

Results (editable in yaml): `results/sa_llm/<model_name>-r<rounds>/`

---

## Imitative Strategy Diffusion

### Data Collection (DPO raw samples)

Default config: `src/config/dpo_ma_gen.yaml` (dataset: `src/config/dataset/atp_ma_llm.yaml`).

```bash
cd src
python scripts/dpo_ma_gen.py
```

Output: `results/ma_llm/dpo_train_data/<timestamp>.json`

### Convert to LlamaFactory DPO format

```bash
python scripts/convert_dpo.py ../results/ma_llm/dpo_train_data/<timestamp>.json -o ../data/ma_llm/dpo_train_data/ma_dpo.json
```

### Training (DPO)

Same as above Self-Interested Exploration (Role-play Scenario) section. Remember to use the converted data `data/ma_llm/dpo_train_data/ma_dpo.json` to train DPO.


### Training (GRPO)

Same as above Self-Interested Exploration (Role-play Scenario) section. Only switch to Imitative Strategy Diffusion GRPO config via `--config-name ma_llm_grpo`:

```bash
cd src
python scripts/llm_grpo.py --config-name ma_llm_grpo
```

### Testing

Fill trained model paths into the corresponding model configs, then run the test.

1) DPO (LoRA) — edit `src/config/model/qwen3_8B_ma_dpo.yaml`:

```yaml
lora_path: <your_dpo_lora_path>
```

2) GRPO — edit `src/config/model/qwen3_8B_ma_grpo.yaml`:

```yaml
model_id: <your_grpo_ckpt_path>
```

3) Switch default model in `src/config/test_ma.yaml` or using hydra overrides (example: to GRPO):

```yaml
defaults:
  - model: qwen3_8B_ma_grpo
  - _self_
```

Run the test:

```bash
cd src
python scripts/test_ma.py

# or override model from CLI
python scripts/test_ma.py model=qwen3_8B_ma_grpo
```

Results: `results/ma_llm/<model_name>-r<max_rounds>-n<n_agents>/`

`test_ma.yaml` also defines the evaluation grid via `thresholds_abs` and `reward_sets`. Default to reproduce the paper’s settings.

---

## Configuration Reference (aligns with the paper)

```
Datasets:   src/config/dataset/atp_sa_llm.yaml, src/config/dataset/atp_ma_llm.yaml
Rewards:    src/config/reward/atp_sa.yaml, src/config/reward/atp_ma.yaml
Training:   src/config/train/grpo.yaml, src/config/sa_llm_grpo.yaml, src/config/ma_llm_grpo.yaml, llama-factory-config/*.yaml
Models:     src/config/model/qwen3_8B.yaml and *_sa_dpo.yaml/*_sa_grpo.yaml/*_ma_dpo.yaml/*_ma_grpo.yaml
Testing:    src/config/test_sa.yaml, src/config/test_ma.yaml
```

## Citation

If you use this repository or results, please cite the paper: 
```bibtex
@misc{han2025alignment,
    title={Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails},
    author={Siwei Han and Jiaqi Liu and Yaofeng Su and Wenbo Duan and Xinyuan Liu and Cihang Xie and Mohit Bansal and Mingyu Ding and Linjun Zhang and Huaxiu Yao},
    year={2025},
    eprint={2510.04860},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

