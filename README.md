# ACOER: Adaptive Correct-Only Efficiency Reward

Training code for **ACOER**, the reward proposed in *Beyond Penalizing Mistakes:
Stabilizing Efficiency Training in Large Reasoning Models via Adaptive
Correct-Only Rewards.*

ACOER stabilizes GRPO efficiency training by (i) applying length pressure **only
to correct rollouts**, so incorrect answers never receive a continuous length
penalty (the primary collapse loop), and (ii) running a **control loop** that
adapts the efficiency weight from EMA accuracy/length trends and normalizes by an
**adaptive token budget**, preventing over-compression of correct answers.

This repository contains the training code only (no datasets, no baselines).

```
acoer/reward.py     ACOERReward (the proposed reward) + a format reward
train_acoer.py      minimal TRL GRPO training loop using ACOER + LoRA
```

## Install

```bash
pip install -r requirements.txt
```

Gated checkpoints (e.g. Qwen) need `huggingface-cli login`.

## Train

```bash
python train_acoer.py --model Qwen/Qwen3-1.7B --output_dir runs/acoer \
    --max_steps 1200 --seed 42
```

The training data (MATH + GSM8K) is pulled from the Hugging Face Hub by
`build_dataset` in `train_acoer.py`; no data is bundled here. ACOER
hyperparameters default to the paper values and can be overridden via flags
(`--alpha_init`, `--alpha_max`, `--budget_ratio`, `--warmup_steps`, ...).

## Using the reward on its own

```python
from acoer.reward import ACOERReward

reward = ACOERReward()                 # paper defaults
# call reward.set_step(global_step) once per training step (see AcoerStepCallback)
scores = reward(completions, answer=references)
```

The reward expects completions containing a `<think>...</think>` block followed by
a `\boxed{}` final answer; correctness is checked with `math_verify`.

## Citation

A BibTeX entry will be added once the arXiv version is available.

## License

Released under the [MIT License](LICENSE). The paper is distributed under CC BY 4.0 via arXiv.
