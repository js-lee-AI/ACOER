"""ACOER: Adaptive Correct-Only Efficiency Reward.

Reward for stabilizing efficiency training of large reasoning models under GRPO.
Length pressure is applied ONLY to correct rollouts (so incorrect answers never
receive a continuous length penalty), and a global control loop adapts the
efficiency weight (alpha) from EMA accuracy/length trends, normalizing by an
adaptive token budget.

    correct:   1.0 + alpha_t * f(efficiency)
    incorrect: 0.0
"""

import math
import re
import logging

logger = logging.getLogger(__name__)


# --- Answer extraction & verification ---

def extract_boxed_answer(text: str):
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def extract_answer_tag(text: str):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_predicted_answer(text: str):
    return extract_boxed_answer(text) or extract_answer_tag(text)


def verify_answer(predicted: str, reference: str) -> bool:
    """Verify predicted vs reference, using math_verify when available."""
    if not predicted or not reference:
        return False
    try:
        from math_verify import parse, verify
        parsed_pred = parse(f"\\boxed{{{predicted}}}")
        parsed_ref = parse(f"\\boxed{{{reference}}}")
        if parsed_pred and parsed_ref:
            return bool(verify(parsed_pred, parsed_ref))
    except ImportError:
        pass
    except Exception:
        pass
    return predicted.strip().lower().replace(" ", "") == reference.strip().lower().replace(" ", "")


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking_block(text: str) -> str:
    match = THINK_PATTERN.search(text)
    return match.group(1) if match else ""


def _get_completion_text(completion) -> str:
    """Normalize TRL completions (string or list-of-message-dicts) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        texts = [m["content"] for m in completion if isinstance(m, dict) and "content" in m]
        return "\n".join(texts)
    return str(completion)


# --- Format reward (companion signal: <think>...</think> + \boxed{}) ---

def format_reward_thinking(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        has_think = bool(THINK_PATTERN.search(text))
        has_answer = "\\boxed{" in text
        rewards.append(1.0 if (has_think and has_answer) else 0.0)
    return rewards


# --- ACOER reward ---

class ACOERReward:
    """Control-loop based correct-only efficiency reward.

    Tracks an EMA of batch accuracy and correct-answer thinking length. When
    accuracy is stable/improving it increases alpha (more efficiency pressure);
    when accuracy drops beyond a threshold it backs off. The normalization
    budget B is kept just below the current average correct length, so the
    signal is a global scalar rather than a per-sample adjustment, which
    sidesteps GRPO advantage cancellation.
    """

    def __init__(
        self,
        alpha_init: float = 0.02,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        alpha_up: float = 1.02,
        alpha_down: float = 0.95,
        ema_span: int = 50,
        check_window: int = 100,
        acc_drop_threshold: float = 0.02,
        budget_ratio: float = 0.85,
        min_budget: int = 512,
        mode: str = "log",
        log_scale: float = 5.0,
        max_thinking_tokens: int = 4096,
        warmup_steps: int = 200,
    ):
        self.alpha = alpha_init
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.ema_span = ema_span
        self.check_window = check_window
        self.acc_drop_threshold = acc_drop_threshold
        self.budget_ratio = budget_ratio
        self.min_budget = min_budget
        self.mode = mode
        self.log_scale = log_scale
        self._log_normalizer = math.log(1.0 + log_scale) if mode == "log" else 1.0
        self.max_thinking_tokens = max_thinking_tokens
        self._max_thinking_chars = max_thinking_tokens * 4
        self.warmup_steps = warmup_steps

        self._current_step = 0
        self._ema_decay = 2.0 / (ema_span + 1)

        self._ema_acc = None
        self._ema_correct_len = None  # in chars

        self._acc_history = []
        self._len_history = []

        self._budget_chars = max_thinking_tokens * 4

        self.__name__ = "acoer_reward"

    def set_step(self, step: int):
        """Update current training step (call from a trainer step callback)."""
        self._current_step = step

    def _update_ema(self, batch_acc: float, batch_correct_len: float):
        if self._ema_acc is None:
            self._ema_acc = batch_acc
            self._ema_correct_len = batch_correct_len
        else:
            d = self._ema_decay
            self._ema_acc = d * batch_acc + (1 - d) * self._ema_acc
            self._ema_correct_len = d * batch_correct_len + (1 - d) * self._ema_correct_len

        self._acc_history.append((self._current_step, self._ema_acc))
        self._len_history.append((self._current_step, self._ema_correct_len))

        max_keep = self.check_window * 2
        if len(self._acc_history) > max_keep:
            self._acc_history = self._acc_history[-max_keep:]
            self._len_history = self._len_history[-max_keep:]

    def _adapt_alpha(self):
        if self._current_step < self.warmup_steps:
            return
        if len(self._acc_history) < self.check_window:
            return

        old_acc = self._acc_history[-self.check_window][1]
        cur_acc = self._ema_acc

        acc_drop = old_acc - cur_acc
        if acc_drop > self.acc_drop_threshold:
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_down)
        else:
            self.alpha = min(self.alpha_max, self.alpha * self.alpha_up)

        if self._ema_correct_len is not None and self._ema_correct_len > 0:
            self._budget_chars = max(self.min_budget * 4, self._ema_correct_len * self.budget_ratio)

    def __call__(self, completions, answer=None, **kwargs):
        if answer is None:
            return [0.0] * len(completions)

        rewards = []
        correct_lens = []
        n_correct = 0

        for completion, ref in zip(completions, answer):
            text = _get_completion_text(completion)
            predicted = extract_predicted_answer(text)
            correct = verify_answer(predicted, ref) if predicted else False

            if not correct:
                rewards.append(0.0)
                continue

            n_correct += 1
            thinking = extract_thinking_block(text)
            thinking_chars = len(thinking) if thinking else 0
            correct_lens.append(thinking_chars)

            norm_len = min(thinking_chars / self._budget_chars, 1.0)
            if self.mode == "log":
                efficiency = math.log(1.0 + self.log_scale * (1.0 - norm_len)) / self._log_normalizer
            else:
                efficiency = 1.0 - norm_len

            rewards.append(1.0 + self.alpha * efficiency)

        batch_acc = n_correct / len(completions) if completions else 0.0
        batch_correct_len = (sum(correct_lens) / len(correct_lens)) if correct_lens else 0.0
        self._update_ema(batch_acc, batch_correct_len)
        self._adapt_alpha()

        return rewards

    @property
    def stats(self) -> dict:
        return {
            "acoer/alpha": self.alpha,
            "acoer/ema_acc": self._ema_acc or 0.0,
            "acoer/ema_correct_len_chars": self._ema_correct_len or 0.0,
            "acoer/budget_chars": self._budget_chars,
        }
