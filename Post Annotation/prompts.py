# datara_ai/annotation/prompts.py
from typing import Optional, List

def frame_label_prompt(
    allowed_parts: Optional[List[str]] = None,
    allowed_tools: Optional[List[str]] = None,
    allowed_actions: Optional[List[str]] = None,
) -> str:
    taxonomy = ""
    if allowed_parts:
        taxonomy += f"\nAllowed parts labels: {allowed_parts}"
    if allowed_tools:
        taxonomy += f"\nAllowed tools labels: {allowed_tools}"
    if allowed_actions:
        taxonomy += f"\nAllowed actions labels: {allowed_actions}"

    return f"""
You are annotating a single manufacturing frame.

Return ONLY valid JSON exactly in this schema:
{{
  "parts": ["..."],
  "tools": ["..."],
  "actions": ["..."]
}}

Rules:
- Use short, specific labels (snake_case ok).
- Use 1–5 labels per field.
- No extra keys, no commentary, no markdown.
{taxonomy}
""".strip()