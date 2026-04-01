# SmolVLA Desktop Agent (WIP)

This project is an early-stage Vision-Language-Action (VLA) desktop agent built using SmolVLA from the HuggingFace + LeRobot ecosystem.

The goal is to create an agent that can:
- Take a screenshot of a computer screen
- Interpret a natural language instruction
- Predict an action
- Execute that action locally

---

## Overview

This system replaces a previous Lingbot-based prototype with a cleaner, modular architecture built around SmolVLA.

The agent runs a continuous loop:

Screenshot → Model → Action Vector → Decode → Validate → Execute → Repeat

---

## Motivation

The previous system used Lingbot VLA, which worked as a proof-of-concept but had several limitations:
- Custom, difficult-to-extend architecture
- Limited integration with modern tooling
- No clear path for training and scaling

SmolVLA was chosen because:
- It is part of the HuggingFace + LeRobot ecosystem
- Easier to run locally
- Better support for fine-tuning
- Aligns with modern VLA system design

---

## Project Structure

smolvla_desktop_agent/

├── app.py  
│   Main loop that runs the agent  

├── requirements.txt  
│   Python dependencies  

└── agent/  
    ├── actions.py  
    │   Structured action definitions (CLICK, TYPE, etc.)  

    ├── capture.py  
    │   Screenshot capture using MSS  

    ├── decoder.py  
    │   Converts model output → structured actions  

    ├── executor.py  
    │   Executes actions locally using pyautogui  

    ├── validator.py  
    │   Ensures actions are safe and valid  

    └── policy.py  
        Loads and runs SmolVLA using LeRobot  

---

## Setup

### 1. Create a virtual environment (Windows)

py -m venv .venv  
.\.venv\Scripts\Activate.ps1  

If activation is blocked:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  
.\.venv\Scripts\Activate.ps1  

---

### 2. Install dependencies

python -m pip install --upgrade pip  
python -m pip install -r requirements.txt  

---

## Running the Agent

### Dry run (recommended)

python app.py --model_id <smolvla_model> --dry_run  

This will print predicted actions without executing them.

---

### Full execution

python app.py --model_id <smolvla_model>  

---

## Current Status

This is a work in progress.

## Key Design Insight

A major realization during development:

The hard part is not running the model — it is defining and training the action space.

SmolVLA outputs continuous vectors, not structured actions.  
This system introduces a decoder layer to map model outputs into:

CLICK(x, y)  
TYPE(text)  
PRESS(key)  
SCROLL(amount)  
WAIT(seconds)  

This representation will be critical for training.

---

## Notes

- This system is designed for local experimentation  
- Not safe for unrestricted execution yet  
- Use --dry_run when testing new models  

---

## Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- LeRobot (SmolVLA)  
- MSS (screen capture)  
- PyAutoGUI (automation)  

---