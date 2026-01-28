import json
from pathlib import Path

LANG="de"

INPUT_JSON=f"prompts_{LANG}.json"
OUTPUT_JSON=f"prompts_{LANG}.json"
AUDIO_ROOT = Path(f"audio_prompts/{LANG}")
GENDERS = {
    "female": ["female1", "female2"],
    "male": ["male1", "male2"],
}

def update_json_with_audio_paths(data):
    for task, styles in data.items():
        for style, prompts in styles.items():
            for idx, prompt in enumerate(prompts, start=1):
                # Reset (optional, but safe)
                prompt["female_rec"] = []
                prompt["male_rec"] = []

                for gender_key, gender_variants in GENDERS.items():
                    for gender_variant in gender_variants:
                        filename = f"{LANG}_{gender_variant}_{task}_{style}_{idx}.wav"
                        file_path = AUDIO_ROOT / filename

                        if file_path.exists():
                            prompt[f"{gender_key}_rec"].append(f"prompts/{str(file_path)}")

    return data

# ---- usage ----
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)


data = update_json_with_audio_paths(data)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
