#!/usr/bin/env python3
"""Minimal GRPO training with the ACOER reward (the method proposed in the paper).

Single method, no baselines. Trains a hybrid reasoning model with TRL GRPO using
ACOERReward plus a format reward, with LoRA. The adaptive schedule is driven by a
step callback that feeds the global training step into the reward.

Example:
    python train_acoer.py --model Qwen/Qwen3-1.7B --output_dir runs/acoer \
        --max_steps 1200 --seed 42

Requires: torch, transformers, trl, peft, datasets, math_verify (and a GPU).
Gated checkpoints need `huggingface-cli login`.
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from acoer.reward import ACOERReward, format_reward_thinking

_PROMPT = ("Solve the following math problem. Show your reasoning, then put your "
           "final answer in \\boxed{{}}.\n\n{q}")


def build_dataset(seed: int = 42):
    """MATH + GSM8K formatted for thinking models as {prompt, answer}."""
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    import re
    math_ds = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", name=s, split="train") for s in subjects
    ])
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")

    def fmt_math(ex):
        sol = ex.get("solution", "")
        m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", sol)
        ans = m[-1].strip() if m else sol
        return {"prompt": [{"role": "user", "content": _PROMPT.format(q=ex["problem"])}], "answer": ans}

    def fmt_gsm(ex):
        a = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
        return {"prompt": [{"role": "user", "content": _PROMPT.format(q=ex["question"])}], "answer": a}

    math_fmt = math_ds.map(fmt_math, remove_columns=math_ds.column_names)
    gsm_fmt = gsm_ds.map(fmt_gsm, remove_columns=gsm_ds.column_names)
    return concatenate_datasets([math_fmt, gsm_fmt]).shuffle(seed=seed)


class AcoerStepCallback(TrainerCallback):
    """Feed the global step into the reward so the control loop can adapt."""

    def __init__(self, reward):
        self.reward = reward

    def on_step_begin(self, args, state, control, **kwargs):
        self.reward.set_step(state.global_step)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id or path")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_generations", type=int, default=16, help="GRPO group size")
    ap.add_argument("--max_completion_length", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--no_lora", action="store_true")
    # ACOER hyperparameters (paper defaults; used without tuning)
    ap.add_argument("--alpha_init", type=float, default=0.02)
    ap.add_argument("--alpha_min", type=float, default=0.01)
    ap.add_argument("--alpha_max", type=float, default=0.5)
    ap.add_argument("--alpha_up", type=float, default=1.02)
    ap.add_argument("--alpha_down", type=float, default=0.95)
    ap.add_argument("--ema_span", type=int, default=50)
    ap.add_argument("--check_window", type=int, default=100)
    ap.add_argument("--acc_drop_threshold", type=float, default=0.02)
    ap.add_argument("--budget_ratio", type=float, default=0.85)
    ap.add_argument("--min_budget", type=int, default=512)
    ap.add_argument("--warmup_steps", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)

    reward = ACOERReward(
        alpha_init=args.alpha_init, alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        alpha_up=args.alpha_up, alpha_down=args.alpha_down, ema_span=args.ema_span,
        check_window=args.check_window, acc_drop_threshold=args.acc_drop_threshold,
        budget_ratio=args.budget_ratio, min_budget=args.min_budget,
        warmup_steps=args.warmup_steps,
    )
    reward_funcs = [reward, format_reward_thinking]

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=0.001,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        seed=args.seed,
        data_seed=args.seed,
    )

    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(seed=args.seed)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(AcoerStepCallback(reward))

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
