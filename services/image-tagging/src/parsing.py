# datara_ai/annotation/parsing.py
import json

def extract_first_json(text: str) -> dict:
    """
    Extract the first JSON object from a model response.
    Raises ValueError if no JSON object found or invalid JSON.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON found in output:\n{text}")

    json_str = text[start:end+1]
    return json.loads(json_str)
