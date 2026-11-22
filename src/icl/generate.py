"""Generation behind a small interface, so tests inject a fake and the Mac never needs vllm.

Decoding for the main ICL arm: greedy (temperature 0), at most `max_new_tokens` tokens, free
generation, no thinking. This is the same decoding as the fine-tuned arm
(src/train/task_spec.py: greedy, 32 new tokens) and the output is parsed by the same parser.

Constrained decoding (`choices` given) is NOT part of the main arm. It exists only as the
separate, labelled SLM robustness check of protocol sections 5 and 6; callers must record it.

vLLM notes. The server has vllm 0.13.0 (protocol section 7). Nothing below was run against it
(the Mac has no vllm); every vLLM argument name that was not checked against the installed
package is marked [U] and must be confirmed on the server before the first real run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence

from src.api.prompts import Prompt
from src.icl.render import chat_messages


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    finish_reason: Optional[str]


class Generator(Protocol):
    name: str

    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int,
                 choices: Optional[Sequence[str]] = None) -> List[Generation]:
        """One Generation per prompt, in order. `choices` switches on constrained decoding
        (robustness flag only); None means free generation."""


class VLLMGenerator:
    """Offline vLLM engine (`vllm.LLM`), imported lazily. Chat template of the model is applied by vLLM."""

    def __init__(self, model: str, *, max_model_len: int = 8192, gpu_memory_utilization: float = 0.85,
                 dtype: str = "auto", seed: int = 0, stop_token_ids: Optional[Sequence[int]] = None,
                 llm: Any = None):
        self.name = model
        self._model = model
        self._kw = dict(max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                        dtype=dtype, seed=seed)  # [U] argument names as in LLM(...) of 0.13.0
        self._stop_token_ids = list(stop_token_ids) if stop_token_ids else None
        self._llm = llm

    def _engine(self):
        if self._llm is None:
            from vllm import LLM  # lazy: only the GPU server has it

            self._llm = LLM(model=self._model, **self._kw)
        return self._llm

    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int,
                 choices: Optional[Sequence[str]] = None) -> List[Generation]:
        from vllm import SamplingParams

        kw: dict = dict(temperature=0.0, max_tokens=max_new_tokens)  # temperature 0 = greedy
        if self._stop_token_ids:
            kw["stop_token_ids"] = self._stop_token_ids  # [U] Phi-4-mini closes turns with <|end|>
        if choices is not None:
            from vllm.sampling_params import StructuredOutputsParams  # [U] name and location in 0.13.0

            kw["structured_outputs"] = StructuredOutputsParams(choice=list(choices))  # [U]
        params = SamplingParams(**kw)
        msgs = [chat_messages(p) for p in prompts]
        # [U] LLM.chat(..., chat_template_kwargs=...) in 0.13.0. enable_thinking=False is a no-op for
        # Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct (no thinking mode; verify on the server that the
        # rendered text has no <think> block), and keeps thinking off if a hybrid model is used.
        outs = self._engine().chat(msgs, params, use_tqdm=False,
                                   chat_template_kwargs={"enable_thinking": False})
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(Generation(text=c.text, input_tokens=len(o.prompt_token_ids),
                                  output_tokens=len(c.token_ids), finish_reason=c.finish_reason))
        return res
