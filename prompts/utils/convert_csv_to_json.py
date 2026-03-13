import csv
import json

INPUT_CSV = "Prompts to Record (DOWIS) - SV.csv"
OUTPUT_JSON = "prompts_sv.json"

def normalize_key(value):
    return value.strip().lower()

def normalize_prompt(text):
    if not text:
        return text

    # Common quote characters (ASCII + Unicode)
    quotes = "\"'“”‘’„"

    # Strip whitespace first
    text = text.strip()

    # Remove leading quotes
    while text and text[0] in quotes:
        text = text[1:].lstrip()

    # Remove trailing quotes
    while text and text[-1] in quotes:
        text = text[:-1].rstrip()

    return text


data = {}

current_task = None
current_style = None

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        task_type = (row["Task Type"] or "").strip()
        if "Ready" in task_type: task_type = ""
        prompt = normalize_prompt(row["Prompt"] or "")

        # Detect task headers like "ASR:" or "ST:"
        if task_type.endswith(":"):
            current_task = normalize_key(task_type[:-1])
            data[current_task] = {}
            current_style = None
            continue

        # Skip separators / empty rows
        if not prompt and not task_type:
            continue

        # Detect new style (Basic, Formal, etc.)
        if task_type:
            current_style = normalize_key(task_type)
            data[current_task].setdefault(current_style, [])

        # Add prompt entry
        if current_task and current_style and prompt:
            data[current_task][current_style].append({
                "text": prompt,
                "female_rec": [],
                "male_rec": []
            })

# Write JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_JSON}")
