"""Interactive character and user-gender selection for the Persona Chatbot.

Call select_character() and select_user_gender() at the top of the notebook
before loading the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.custom_ai import (
    create_custom_ai,
    edit_custom_character,
    ensure_custom_dialogue_file,
    save_custom_character,
)


_GENDER_OPTIONS: list[tuple[str, str]] = [
    ("male",    "Male   (he/him)"),
    ("female",  "Female (she/her)"),
    ("neutral", "Prefer not to say"),
]


def load_characters(
    config_path: str | Path = "config/characters.json",
) -> list[dict[str, Any]]:
    """Return the character list from characters.json."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)["characters"]


def get_builtin_characters(characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in characters if not c.get("is_custom")]


def get_custom_characters(characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in characters if c.get("is_custom")]


def filter_characters(
    characters: list[dict[str, Any]],
    gender: str | None = None,
    custom_only: bool | None = None,
) -> list[dict[str, Any]]:
    filtered = characters
    if gender is not None:
        filtered = [c for c in filtered if c.get("gender") == gender]
    if custom_only is True:
        filtered = get_custom_characters(filtered)
    elif custom_only is False:
        filtered = get_builtin_characters(filtered)
    return filtered


def _display_menu(characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    print("\n" + "=" * 58)
    print("      ✦  PERSONA CHATBOT  —  CHARACTER SELECT  ✦")
    print("=" * 58)

    ordered: list[dict[str, Any]] = []
    sections = [
        ("Built-in Female Characters", filter_characters(characters, gender="female", custom_only=False)),
        ("Built-in Male Characters", filter_characters(characters, gender="male", custom_only=False)),
        ("Custom Characters", get_custom_characters(characters)),
    ]

    for title, section_chars in sections:
        if section_chars:
            print(f"\n  -- {title} " + "-" * max(1, 43 - len(title)))
        for c in section_chars:
            ordered.append(c)
            dere = c.get("dere_type", "")
            marker = "custom" if c.get("is_custom") else dere
            print(f"  [{len(ordered)}]  {c['display_name']:<34}  {marker}")

    print("\n" + "=" * 58)
    return ordered


def select_character(
    config_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    """Prompt the user to pick a character; return its config dict."""
    characters = load_characters(config_path)
    displayed_characters = _display_menu(characters)

    while True:
        try:
            raw = input(f"  Enter number [1-{len(displayed_characters)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(displayed_characters):
                chosen = displayed_characters[idx]
                dere   = chosen.get("dere_type", "")
                print(f"\n  ✓  Selected  →  {chosen['display_name']}  [{dere}]\n")
                return chosen
            print(f"  Please enter a number between 1 and {len(displayed_characters)}.")
        except (ValueError, EOFError):
            print("  Invalid input — please enter a number.")


def select_or_create_character(
    config_path: str | Path = "config/characters.json",
) -> dict[str, Any]:
    """Choose an existing character or create/edit a custom character."""
    print("=" * 58)
    print("  V1.3 Character Mode")
    print("=" * 58)
    print("  [1] Use existing character")
    print("  [2] Create custom AI")
    print("  [3] Load custom AI")
    print("  [4] Edit custom AI")
    print()

    while True:
        raw = input("  Enter number [1-4]: ").strip()
        if raw in {"1", "2", "3", "4"}:
            break
        print("  Invalid input - please enter 1, 2, 3, or 4.")

    characters = load_characters(config_path)

    if raw == "1":
        builtins = get_builtin_characters(characters)
        return select_character_from_list(builtins)

    if raw == "2":
        existing_ids = [c["id"] for c in characters]
        character = create_custom_ai(existing_ids=existing_ids)
        saved = save_custom_character(character, characters_path=config_path)
        dialogue_path = ensure_custom_dialogue_file(saved)
        print(f"\n  Created {saved['display_name']} with dialogue file {dialogue_path}\n")
        return saved

    custom_chars = get_custom_characters(characters)
    if not custom_chars:
        print("  No custom characters found. Create one first.")
        existing_ids = [c["id"] for c in characters]
        character = create_custom_ai(existing_ids=existing_ids)
        return save_custom_character(character, characters_path=config_path)

    chosen = select_character_from_list(custom_chars)
    if raw == "3":
        return chosen

    print("  Leave a field blank to keep the current value.")
    updates: dict[str, Any] = {}
    for field in [
        "display_name",
        "personality",
        "speech_style",
        "background",
        "relationship_mode",
        "language_style",
        "response_length",
    ]:
        current = chosen.get(field, "")
        value = input(f"  {field} [{current}]: ").strip()
        if value:
            updates[field] = value
    return edit_custom_character(chosen["id"], updates, characters_path=config_path)


def select_character_from_list(characters: list[dict[str, Any]]) -> dict[str, Any]:
    displayed_characters = _display_menu(characters)
    while True:
        try:
            raw = input(f"  Enter number [1-{len(displayed_characters)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(displayed_characters):
                chosen = displayed_characters[idx]
                print(f"\n  ✓  Selected  →  {chosen['display_name']}\n")
                return chosen
            print(f"  Please enter a number between 1 and {len(displayed_characters)}.")
        except (ValueError, EOFError):
            print("  Invalid input - please enter a number.")


def select_user_gender() -> str:
    """Prompt the user to state their gender; return 'male', 'female', or 'neutral'."""
    print("=" * 58)
    print("  Your gender influences how your character interacts.")
    print("=" * 58)
    for i, (_, label) in enumerate(_GENDER_OPTIONS, 1):
        print(f"  [{i}]  {label}")
    print()

    while True:
        try:
            raw = input(f"  Enter number [1–{len(_GENDER_OPTIONS)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(_GENDER_OPTIONS):
                key, label = _GENDER_OPTIONS[idx]
                print(f"\n  ✓  Set to  →  {label}\n")
                return key
            print(f"  Please enter a number between 1 and {len(_GENDER_OPTIONS)}.")
        except (ValueError, EOFError):
            print("  Invalid input — please enter a number.")
