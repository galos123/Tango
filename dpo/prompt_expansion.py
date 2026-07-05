#!/usr/bin/env python3
"""
prompt_expansion.py — Turn a short audio caption into a more detailed one
using a HuggingFace instruction-tuned LLM.

Two "prompts" are sent to the LLM for every caption, as chat turns:
  1. A system instruction telling the model to rewrite the caption with more
     acoustic/temporal/environmental detail.
  2. The actual sound caption, as the user turn.

The LLM's reply is the detailed caption used to build a preference pair in
`build_preference_dataset.py` (see that file for how chosen/rejected are
assigned).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DETAIL_INSTRUCTION = (
    "You are a professional sound designer writing prompts for a text-to-audio "
    "generation model. Rewrite the given short audio caption as a single, more "
    "detailed and vivid description of the same sound: mention the sound "
    "source(s), acoustic qualities (pitch, texture, loudness), temporal "
    "structure (onset, duration, repetition) and the environment/space it "
    "happens in. Do not invent unrelated sounds. Respond with ONLY the "
    "rewritten caption as one sentence — no preamble, no quotes."
)


class PromptExpander:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        device: str | None = None,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def expand(self, caption: str) -> str:
        """Send the (instruction, caption) prompt pair to the LLM and return
        the detailed rewrite. Falls back to the original caption if the model
        produces an empty response."""
        messages = [
            {"role": "system", "content": DETAIL_INSTRUCTION},  # prompt 1: instruction
            {"role": "user", "content": caption},                # prompt 2: actual sound prompt
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        reply = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip().strip('"')
        return reply or caption

    def expand_batch(self, captions: list[str]) -> list[str]:
        return [self.expand(c) for c in captions]
