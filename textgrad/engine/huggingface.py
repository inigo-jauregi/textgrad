"""Local HuggingFace `transformers` engine for TextGrad / REVOLVE.

Adapted from the REVOLVE repository (https://github.com/peiyance/revolve),
`revolve/engine/llama3_1.py` (`ChatLlama3_1`), and generalised so that any
HuggingFace causal-LM checkpoint can be loaded and served locally as a TextGrad
`EngineLM`. It has extra handling for the Qwen3 family (in particular the
`enable_thinking` chat-template switch and the `<think>...</think>` reasoning
blocks these models emit).

Requirements (not installed by default):
    pip install "transformers>=4.51.0" torch accelerate

Note: Qwen3 support was added in transformers 4.51.0, so an older version will
fail to load `Qwen/Qwen3-8B`.
"""

import os
import platformdirs

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError(
        "To use the local HuggingFace engine, install the dependencies with "
        '`pip install "transformers>=4.51.0" torch accelerate`.'
    )

from typing import List, Union
from .base import EngineLM, CachedEngine


class ChatHuggingFace(EngineLM, CachedEngine):
    """Serve a local HuggingFace causal LM as a TextGrad engine.

    Example
    -------
    >>> from textgrad.engine.huggingface import ChatHuggingFace
    >>> engine = ChatHuggingFace(model_string="Qwen/Qwen3-8B", enable_thinking=False)
    >>> print(engine.generate("What is the capital of France?"))
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "Qwen/Qwen3-8B",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device_map: str = "auto",
        torch_dtype="auto",
        enable_thinking: bool = None,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = None,
        trust_remote_code: bool = True,
        **model_kwargs,
    ):
        """
        :param model_string: HF hub id or a local path to the checkpoint.
        :param system_prompt: default system prompt used when none is provided.
        :param device_map: passed to `from_pretrained` (e.g. "auto", "cuda", "cpu", "mps").
        :param torch_dtype: dtype for the weights ("auto", torch.bfloat16, ...).
        :param enable_thinking: Qwen3-specific chat-template switch. When None the
            argument is not forwarded to the tokenizer (use this for non-Qwen models).
            Set to False for fast, clean, easy-to-parse optimizer outputs, or True to
            let the model reason inside <think>...</think> blocks (which are stripped
            from the returned text).
        :param max_new_tokens: generation cap.
        :param temperature / top_p / top_k: sampling parameters. For Qwen3 in
            non-thinking mode the recommended values are temperature=0.7, top_p=0.8,
            top_k=20; in thinking mode temperature=0.6, top_p=0.95, top_k=20.
        :param do_sample: whether to sample. Defaults to (temperature > 0). Set
            temperature=0 (or do_sample=False) for greedy/deterministic decoding.
        :param trust_remote_code: forwarded to `from_pretrained`.
        :param model_kwargs: any extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained`.
        """
        root = platformdirs.user_cache_dir("textgrad")
        model_name = model_string.split("/")[-1]
        cache_path = os.path.join(root, f"cache_hf_{model_name}.db")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample if do_sample is not None else (temperature > 0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_string, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_string,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        self.model.eval()

    def _apply_chat_template(self, messages):
        """Render messages with the model's chat template.

        Only forwards `enable_thinking` when the caller asked for it, so the
        template still works for models that don't understand that kwarg.
        """
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.enable_thinking is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, enable_thinking=self.enable_thinking, **kwargs
                )
            except TypeError:
                # Template doesn't accept enable_thinking (non-Qwen3 model).
                pass
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove a leading Qwen3 <think>...</think> reasoning block, if present."""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()

    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str = None,
        **kwargs,
    ):
        sys_prompt_arg = system_prompt if system_prompt is not None else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + content)
        if cache_or_none is not None:
            return cache_or_none

        messages = []
        if sys_prompt_arg:
            messages.append({"role": "system", "content": sys_prompt_arg})
        messages.append({"role": "user", "content": content})

        text = self._apply_chat_template(messages)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k),
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = self._strip_thinking(response)

        self._save_cache(sys_prompt_arg + content, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)