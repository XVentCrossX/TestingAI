"""Custom AI character and dialogue-example helpers for V1.3.

This module keeps the notebook light: it validates character profiles, writes
custom characters into ``config/characters.json``, and manages beginner-friendly
``USER:`` / ``CHARACTER:`` dialogue examples.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


REQUIRED_CHARACTER_FIELDS = [
    "id",
    "display_name",
    "gender",
    "dere_type",
    "age",
    "personality",
    "speech_style",
    "background",
    "likes",
    "dislikes",
    "greeting_neutral",
    "greeting_male",
    "greeting_female",
    "dialogue_file",
    "is_custom",
]

OPTIONAL_CHARACTER_DEFAULTS: dict[str, Any] = {
    "relationship_mode": "friend",
    "language_style": "english",
    "response_length": "short",
    "boundaries": ["Keep the relationship respectful and non-explicit."],
    "custom_rules": [],
    "example_dialogues": [],
}

VALID_GENDERS = {"female", "male", "neutral", "custom"}


def _read_characters_file(characters_path: str | Path) -> dict[str, Any]:
    path = Path(characters_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("characters", [])
    return data


def _write_characters_file(data: dict[str, Any], characters_path: str | Path) -> None:
    path = Path(characters_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _split_csv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def generate_character_id(name: str, existing_ids: Optional[Iterable[str]] = None) -> str:
    """Return a safe lowercase custom character id derived from a display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        slug = "ai"
    if not slug.startswith("custom_"):
        slug = f"custom_{slug}"

    existing = set(existing_ids or [])
    candidate = slug
    index = 2
    while candidate in existing:
        candidate = f"{slug}_{index}"
        index += 1
    return candidate


def validate_character_schema(character: dict[str, Any]) -> list[str]:
    """Return validation errors for a character dictionary."""
    errors: list[str] = []

    for field in REQUIRED_CHARACTER_FIELDS:
        if field not in character:
            errors.append(f"Missing required field: {field}")

    char_id = str(character.get("id", ""))
    if char_id and not re.fullmatch(r"[a-z0-9_]+", char_id):
        errors.append("Character id must use only lowercase letters, numbers, and underscores.")

    gender = str(character.get("gender", "")).lower()
    if gender and gender not in VALID_GENDERS:
        errors.append(f"Gender must be one of: {', '.join(sorted(VALID_GENDERS))}.")

    try:
        age = int(character.get("age", 0))
        if age <= 0:
            errors.append("Age must be a positive number.")
        relationship_mode = str(character.get("relationship_mode", "friend")).lower()
        if age < 18 and relationship_mode in {"romantic", "partner", "dating"}:
            errors.append("Romantic/partner relationship mode requires a character age of 18 or older.")
    except (TypeError, ValueError):
        errors.append("Age must be an integer.")

    for list_field in ("likes", "dislikes", "boundaries", "custom_rules", "example_dialogues"):
        if list_field in character and not isinstance(character[list_field], list):
            errors.append(f"{list_field} must be a list.")

    dialogue_file = str(character.get("dialogue_file", ""))
    if not dialogue_file:
        errors.append("dialogue_file is required.")

    return errors


def create_custom_ai(
    name: Optional[str] = None,
    gender: Optional[str] = None,
    age: Optional[int] = None,
    relationship_mode: Optional[str] = None,
    personality: Optional[str] = None,
    speech_style: Optional[str] = None,
    background: Optional[str] = None,
    likes: str | Iterable[str] | None = None,
    dislikes: str | Iterable[str] | None = None,
    greeting_neutral: Optional[str] = None,
    greeting_male: Optional[str] = None,
    greeting_female: Optional[str] = None,
    boundaries: str | Iterable[str] | None = None,
    language_style: Optional[str] = None,
    response_length: Optional[str] = None,
    custom_rules: str | Iterable[str] | None = None,
    existing_ids: Optional[Iterable[str]] = None,
    prompt_gender_greetings: bool = True,
) -> dict[str, Any]:
    """Create a custom AI dictionary, prompting for any missing core fields."""
    name = name or _prompt_text("AI name")
    gender = (gender or _prompt_text("AI gender", "neutral")).lower()
    age_value = age if age is not None else int(_prompt_text("AI age", "18"))
    relationship_mode = relationship_mode or _prompt_text("Relationship mode", "friend")

    personality = personality or _prompt_text("Personality")
    speech_style = speech_style or _prompt_text("Speech style")
    background = background or _prompt_text("Background")
    if likes is None:
        likes = _prompt_text("Likes, comma separated", "")
    if dislikes is None:
        dislikes = _prompt_text("Dislikes, comma separated", "")
    greeting_neutral = greeting_neutral or _prompt_text("Greeting for neutral user", f"Hi, I'm {name}.")
    if greeting_male is None:
        greeting_male = (
            _prompt_text("Greeting for male user", greeting_neutral)
            if prompt_gender_greetings
            else greeting_neutral
        )
    if boundaries is None:
        boundaries = _prompt_text(
            "Boundaries, comma separated",
            "Keep the relationship respectful and non-explicit.",
        )
    language_style = language_style or _prompt_text("Language style", "english")
    response_length = response_length or _prompt_text("Response length", "short")
    if custom_rules is None:
        custom_rules = _prompt_text("Custom rules, comma separated", "")
    if greeting_female is None:
        greeting_female = (
            _prompt_text("Greeting for female user", greeting_neutral)
            if prompt_gender_greetings
            else greeting_neutral
        )

    char_id = generate_character_id(name, existing_ids=existing_ids)
    character = {
        "id": char_id,
        "display_name": name,
        "gender": gender,
        "dere_type": "Custom",
        "age": int(age_value),
        "personality": personality,
        "speech_style": speech_style,
        "background": background,
        "likes": _split_csv(likes),
        "dislikes": _split_csv(dislikes),
        "greeting_neutral": greeting_neutral,
        "greeting_male": greeting_male,
        "greeting_female": greeting_female,
        "dialogue_file": f"data/custom_dialogues/dialogues_{char_id}.txt",
        "is_custom": True,
        "relationship_mode": relationship_mode,
        "language_style": language_style,
        "response_length": response_length,
        "boundaries": _split_csv(boundaries) or list(OPTIONAL_CHARACTER_DEFAULTS["boundaries"]),
        "custom_rules": _split_csv(custom_rules),
        "example_dialogues": [],
    }

    errors = validate_character_schema(character)
    if errors:
        raise ValueError("Invalid custom AI profile:\n- " + "\n- ".join(errors))
    return character


def save_custom_character(
    character: dict[str, Any],
    characters_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    """Append or update a custom character in characters.json and ensure its dialogue file."""
    character = {**OPTIONAL_CHARACTER_DEFAULTS, **character}
    character["is_custom"] = bool(character.get("is_custom", True))

    errors = validate_character_schema(character)
    if errors:
        raise ValueError("Invalid custom AI profile:\n- " + "\n- ".join(errors))

    data = _read_characters_file(characters_path)
    chars = data["characters"]
    existing_index = next((i for i, c in enumerate(chars) if c.get("id") == character["id"]), None)
    if existing_index is None:
        chars.append(character)
    else:
        chars[existing_index] = {**chars[existing_index], **character}

    _write_characters_file(data, characters_path)
    ensure_custom_dialogue_file(character)
    return character


def load_custom_character(
    character_id: str,
    characters_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    data = _read_characters_file(characters_path)
    for character in data["characters"]:
        if character.get("id") == character_id:
            return character
    raise KeyError(f"Character not found: {character_id}")


def edit_custom_character(
    character_id: str,
    updates: dict[str, Any],
    characters_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    character = load_custom_character(character_id, characters_path)
    if not character.get("is_custom"):
        raise ValueError("Only custom characters can be edited with edit_custom_character().")
    updated = {**character, **updates}
    return save_custom_character(updated, characters_path)


def ensure_custom_dialogue_file(
    character: dict[str, Any],
    base_dir: str | Path = "data/custom_dialogues",
) -> Path:
    dialogue_path = Path(character.get("dialogue_file", ""))
    if not dialogue_path:
        dialogue_path = Path(base_dir) / f"dialogues_{character['id']}.txt"

    dialogue_path.parent.mkdir(parents=True, exist_ok=True)
    if not dialogue_path.exists():
        display_name = character.get("display_name", character.get("id", "Character"))
        dialogue_path.write_text(
            f"# Custom dialogue examples for {display_name}\n"
            "# Format:\n"
            "# USER: Hello\n"
            "# CHARACTER: Hi. I'm glad you came by.\n\n",
            encoding="utf-8",
        )
    return dialogue_path


def parse_dialogue_examples(text: str) -> list[dict[str, str]]:
    """Parse beginner-friendly USER/CHARACTER dialogue text into pairs."""
    pairs: list[dict[str, str]] = []
    pending_user: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("USER:"):
            pending_user = line.split(":", 1)[1].strip()
        elif upper.startswith("CHARACTER:"):
            reply = line.split(":", 1)[1].strip()
            if pending_user is not None:
                pairs.append({"user": pending_user, "character": reply})
                pending_user = None
        elif ":" in line and pending_user is not None:
            reply = line.split(":", 1)[1].strip()
            pairs.append({"user": pending_user, "character": reply})
            pending_user = None

    return pairs


def validate_dialogue_pairs(pairs: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not pairs:
        errors.append("No valid USER/CHARACTER pairs found.")
    for index, pair in enumerate(pairs, 1):
        if not pair.get("user", "").strip():
            errors.append(f"Pair {index} is missing USER text.")
        if not pair.get("character", "").strip():
            errors.append(f"Pair {index} is missing CHARACTER text.")
    return errors


def append_dialogue_examples(
    dialogue_path: str | Path,
    examples_text: str,
    min_training_pairs: int = 50,
) -> dict[str, Any]:
    pairs = parse_dialogue_examples(examples_text)
    errors = validate_dialogue_pairs(pairs)
    if errors:
        raise ValueError("Invalid dialogue examples:\n- " + "\n- ".join(errors))

    path = Path(dialogue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size > 0:
            f.write("\n")
        for pair in pairs:
            f.write(f"USER: {pair['user']}\n")
            f.write(f"CHARACTER: {pair['character']}\n\n")

    total_pairs = count_dialogue_pairs(path)
    warnings = []
    if total_pairs < min_training_pairs:
        warnings.append(
            f"{total_pairs} dialogue pairs is useful for prompt examples, "
            f"but too small for reliable fine-tuning."
        )
    return {"added": len(pairs), "total_pairs": total_pairs, "warnings": warnings}


def count_dialogue_pairs(dialogue_path: str | Path) -> int:
    path = Path(dialogue_path)
    if not path.exists():
        return 0
    return len(parse_dialogue_examples(path.read_text(encoding="utf-8")))
