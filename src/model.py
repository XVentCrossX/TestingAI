"""Transformer chatbot core.

Loads a small instruction-tuned causal LM from HuggingFace and generates a
reply conditioned on the persona (system prompt), an optional rolling
memory summary, and the recent dialogue history (chat-template messages).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.memory import ConversationMemory
from src.persona import Persona


class PersonaChatbot:
    def __init__(
        self,
        persona: Persona,
        memory: ConversationMemory,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        seed: int = 42,
        device: Optional[str] = None,
        quantization: str = "none",
        save_memory_path: Optional[str | Path] = None,
    ) -> None:
        self.persona = persona
        self.memory  = memory

        self.model_name        = model_name
        self.max_new_tokens    = max_new_tokens
        self.temperature       = temperature
        self.top_p             = top_p
        self.top_k             = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample         = do_sample
        self.seed              = seed
        self.quantization      = quantization
        self.save_memory_path  = Path(save_memory_path) if save_memory_path else None

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantization == "4bit" and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto"
            )
        elif quantization == "8bit" and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.model.eval()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/config.json",
        persona: Optional[Persona] = None,
        memory: Optional[ConversationMemory] = None,
        character_id: str = "mai",
        user_gender: str = "neutral",
    ) -> "PersonaChatbot":
        with Path(config_path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        m = data.get("model", {})

        persona = persona or Persona.from_config(
            config_path, character_id=character_id, user_gender=user_gender
        )
        memory  = memory  or ConversationMemory.from_config(config_path)

        return cls(
            persona=persona,
            memory=memory,
            model_name=m.get("name", "Qwen/Qwen2.5-7B-Instruct"),
            max_new_tokens=m.get("max_new_tokens", 120),
            temperature=m.get("temperature", 0.8),
            top_p=m.get("top_p", 0.9),
            top_k=m.get("top_k", 50),
            repetition_penalty=m.get("repetition_penalty", 1.1),
            do_sample=m.get("do_sample", True),
            seed=m.get("seed", 42),
            quantization=m.get("quantization", "none"),
        )

    def _build_messages(
        self, user_input: str, use_persona: bool = True
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if use_persona:
            system_prompt = self.persona.build_prefix(
                memory_summary=self.memory.summary
            )
            messages.append({"role": "system", "content": system_prompt.strip()})

        for turn in self.memory.get_recent():
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_input})
        return messages

    @torch.no_grad()
    def generate(
        self,
        user_input: str,
        use_persona: bool = True,
        record: bool = True,
    ) -> str:
        messages = self._build_messages(user_input, use_persona=use_persona)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.max_new_tokens,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        if not reply:
            reply = "..."

        if record:
            self.memory.add_turn("user",      user_input)
            self.memory.add_turn("assistant", reply)
            if self.save_memory_path is not None:
                self.memory.save_memory(
                    self.save_memory_path,
                    character_id=self.persona.character_id,
                )

        return reply

    def reset_conversation(self) -> None:
        self.memory.reset()
        torch.manual_seed(self.seed)

    def switch_character(
        self,
        character_id: str,
        config_path: str | Path = "config/config.json",
        user_gender: Optional[str] = None,
    ) -> None:
        ug = user_gender if user_gender is not None else self.persona.user_gender
        self.persona = Persona.from_config(
            config_path, character_id=character_id, user_gender=ug
        )
        self.reset_conversation()
        print(f"Switched to {self.persona.name}  [{self.persona.dere_type}]")

    def update_generation_settings(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> None:
        """Update generation settings without reloading the model."""
        if temperature is not None:
            self.temperature = float(temperature)
        if top_p is not None:
            self.top_p = float(top_p)
        if top_k is not None:
            self.top_k = int(top_k)
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)
        if repetition_penalty is not None:
            self.repetition_penalty = float(repetition_penalty)
        if do_sample is not None:
            self.do_sample = bool(do_sample)
