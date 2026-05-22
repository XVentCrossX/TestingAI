"""Conversation memory with rolling summarization.

Stores turns as a list of ``{"role": ..., "content": ...}`` dicts and, every
``window_size`` turns, condenses older turns into a short summary using a
HuggingFace summarization pipeline. The summary is then injected as context
into subsequent turns through ``Persona.build_prefix``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - transformers may not be installed at import time
    pipeline = None  # type: ignore[assignment]


class ConversationMemory:
    def __init__(
        self,
        window_size: int = 10,
        summarizer_model: str = "sshleifer/distilbart-cnn-12-6",
        summary_max_length: int = 80,
        summary_min_length: int = 20,
        lazy_load: bool = True,
    ) -> None:
        self.window_size = window_size
        self.summarizer_model = summarizer_model
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length

        self.history: list[dict[str, str]] = []
        self.summary: str = ""
        self.turn_count: int = 0

        self._summarizer = None
        if not lazy_load:
            self._load_summarizer()

    def _load_summarizer(self) -> None:
        if self._summarizer is not None:
            return
        if pipeline is None:
            self._summarizer = "fallback"
            return
        try:
            self._summarizer = pipeline("summarization", model=self.summarizer_model)
        except Exception:
            self._summarizer = "fallback"

    @classmethod
    def from_config(cls, config_path: str | Path = "config/config.json") -> "ConversationMemory":
        with Path(config_path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        mem = data.get("memory", {})
        model_cfg = data.get("model", {})
        return cls(
            window_size=mem.get("window_size", 10),
            summarizer_model=model_cfg.get("summarizer", "sshleifer/distilbart-cnn-12-6"),
            summary_max_length=mem.get("summary_max_length", 80),
            summary_min_length=mem.get("summary_min_length", 20),
        )

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content.strip()})
        if role == "user":
            self.turn_count += 1
        if self.turn_count > 0 and self.turn_count % self.window_size == 0 and role == "assistant":
            self.summarize()

    def get_recent(self, n: Optional[int] = None) -> list[dict[str, str]]:
        if n is None:
            n = self.window_size * 2
        return self.history[-n:]

    def format_recent_dialogue(self, persona_name: str, n: Optional[int] = None) -> str:
        """Render the last few turns as a plain dialogue block usable in a prompt."""
        recent = self.get_recent(n)
        lines: list[str] = []
        for turn in recent:
            speaker = "User" if turn["role"] == "user" else persona_name
            lines.append(f"{speaker}: {turn['content']}")
        return "\n".join(lines)

    def summarize(self) -> str:
        """Summarize the full conversation so far and store the result."""
        if not self.history:
            return self.summary

        self._load_summarizer()

        dialogue = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
            for t in self.history
        )

        max_chars = 3500
        if len(dialogue) > max_chars:
            dialogue = dialogue[-max_chars:]

        if self._summarizer == "fallback" or self._summarizer is None:
            self.summary = self._truncation_summary(dialogue)
            return self.summary

        try:
            result = self._summarizer(
                dialogue,
                max_length=self.summary_max_length,
                min_length=self.summary_min_length,
                do_sample=False,
                truncation=True,
            )
            self.summary = result[0]["summary_text"].strip()
        except Exception:
            self.summary = self._truncation_summary(dialogue)
        return self.summary

    def _truncation_summary(self, dialogue: str) -> str:
        """Cheap fallback summary: keep the last few sentences of the dialogue."""
        max_chars = self.summary_max_length * 6
        snippet = dialogue[-max_chars:].replace("\n", " ")
        return snippet.strip()

    def reset(self) -> None:
        self.history.clear()
        self.summary = ""
        self.turn_count = 0

    def save_memory(
        self,
        path: str | Path,
        character_id: Optional[str] = None,
    ) -> Path:
        """Persist the current memory state as JSON."""
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "character_id": character_id,
            "summary": self.summary,
            "history": self.history,
            "turn_count": self.turn_count,
        }
        with memory_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return memory_path

    def load_memory(self, path: str | Path) -> "ConversationMemory":
        """Load a saved memory JSON file into this memory object."""
        memory_path = Path(path)
        with memory_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        self.summary = str(data.get("summary", ""))
        self.history = [
            {
                "role": str(turn.get("role", "")),
                "content": str(turn.get("content", "")),
            }
            for turn in data.get("history", [])
            if isinstance(turn, dict)
        ]
        self.turn_count = int(
            data.get(
                "turn_count",
                sum(1 for turn in self.history if turn.get("role") == "user"),
            )
        )
        return self

    def clear_memory(self, path: str | Path | None = None) -> None:
        """Clear in-memory state and optionally remove a saved memory file."""
        self.reset()
        if path is not None:
            memory_path = Path(path)
            if memory_path.exists():
                memory_path.unlink()
