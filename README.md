# AI Partner V1.3 - Customizable Persona Chatbot

AI Partner is a Google Colab-friendly persona chatbot built around Qwen 2.5
Instruct. V1.3 keeps the six built-in anime-style characters from V1.2 and adds
the first custom AI workflow: create a character in the notebook, save it to
JSON, add dialogue examples, and persist memory across runtime restarts.

RAG knowledge upload, LoRA/QLoRA training, adapter loading, export/import
packages, and advanced evaluation are planned for later milestones.

---

## Features

- Six built-in characters with different personalities and dialogue files.
- Guided custom AI creation from inside `AI_PartnerV1.3.ipynb`.
- Custom characters saved to `config/characters.json`.
- Automatic custom dialogue files in `data/custom_dialogues/`.
- Persistent memory files in `memory/`.
- Notebook cells for persona preview, chat, memory controls, and dialogue
  example entry.
- V1.2 notebook preserved as `AI_PartnerV1.2.ipynb`.

---

## Project Structure

```text
AI-Girlfriend/
├── AI_Girlfriend.ipynb          # Legacy notebook
├── AI_PartnerV1.2.ipynb         # Stable V1.2 notebook
├── AI_PartnerV1.3.ipynb         # V1.3 custom AI notebook
├── requirements.txt
├── config/
│   ├── config.json              # Model, memory, UI, and evaluation settings
│   └── characters.json          # Built-in and custom character definitions
├── data/
│   ├── anime_dialogues.txt
│   ├── dialogues_yuki.txt
│   ├── dialogues_hana.txt
│   ├── dialogues_sora.txt
│   ├── dialogues_rei.txt
│   ├── dialogues_kaito.txt
│   └── custom_dialogues/
├── memory/
└── src/
    ├── character_select.py
    ├── custom_ai.py
    ├── evaluation.py
    ├── memory.py
    ├── model.py
    └── persona.py
```

---

## Quick Start

1. Open `AI_PartnerV1.3.ipynb` in Google Colab.
2. Set Runtime to T4 GPU if you want to load the default 7B model.
3. Run setup and project validation cells. In Colab, the setup cell clones or
   updates `https://github.com/XVentCrossX/TestingAI.git` first if the notebook
   is not already inside a complete repo checkout.
4. Choose an existing character or create a custom AI.
5. Preview the persona prompt.
6. Load the model.
7. Chat, save memory, and add dialogue examples.

For local testing, install the milestone dependencies:

```bash
pip install -r requirements.txt
```

---

## Built-In Characters

| ID | Name | Type | Gender |
|---|---|---|---|
| `mai` | Mai Sakurajima | Tsundere | Female |
| `yuki` | Yuki Amane | Kuudere | Female |
| `hana` | Hana Mizuki | Dandere | Female |
| `sora` | Sora Himari | Deredere | Female |
| `rei` | Rei Shirogane | Himedere | Female |
| `kaito` | Kaito Ryusei | Kuudere | Male |

---

## Custom AI Creator

V1.3 can create a new character from normal notebook inputs:

- Name, gender, and age
- Relationship mode
- Personality, speech style, and background
- Likes and dislikes
- Gender-aware greetings
- Boundaries and custom rules
- Response length and language style

The new character is saved into `config/characters.json`, and a matching
dialogue file is created automatically under `data/custom_dialogues/`.

---

## Persistent Memory

`ConversationMemory` now supports:

- `save_memory(path, character_id=None)`
- `load_memory(path)`
- `clear_memory(path=None)`

The V1.3 notebook stores memory by character in:

```text
memory/<character_id>_memory.json
```

Saved memory contains the character id, summary, recent history, and turn count.

---

## Dialogue Examples

Custom dialogue examples use a beginner-friendly format:

```text
USER: Are you mad at me?
CHARACTER: No. I just wish you told me earlier.

USER: I missed you.
CHARACTER: You say that so casually. I missed you too.
```

The notebook validates and appends these examples to the current character's
custom dialogue file. A small number of examples is useful for prompt/style
testing, but it is not enough for reliable fine-tuning.

---

## Current Limitations

- RAG and knowledge-file retrieval are not implemented in this milestone.
- LoRA/QLoRA training and adapter loading are deferred.
- Export/import packages are deferred.
- BLEU and perplexity evaluation remain available from V1.2, but advanced
  persona, memory, and knowledge evaluation will come later.
- The default 7B model can require a GPU and several minutes to load in Colab.
