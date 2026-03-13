import json
from pathlib import Path
from typing import Optional
from datasets import Dataset, Features, Value, Audio

LANGUAGES  = ["en", "de", "it", "cs", "es", "fr", "hu", "nl", "pt", "ru", "sq", "sv"]
HF_REPO    = "maikezu/dowis"
SCRIPT_DIR = Path(__file__).parent   # audio paths are relative to this dir


def build_rows(lang: str, data: dict) -> list[dict]:
    rows = []
    for task, prompt_types in data.items():
        for prompt_type, prompts in prompt_types.items():
            for prompt in prompts:
                female = [str(SCRIPT_DIR / p) for p in prompt.get("female_rec", [])]
                male   = [str(SCRIPT_DIR / p) for p in prompt.get("male_rec",   [])]
                rows.append({
                    "text_prompt":           prompt["text"],
                    "audio_prompt_female_1": female[0] if len(female) > 0 else None,
                    "audio_prompt_female_2": female[1] if len(female) > 1 else None,
                    "audio_prompt_male_1":   male[0]   if len(male)   > 0 else None,
                    "audio_prompt_male_2":   male[1]   if len(male)   > 1 else None,
                    "language":    lang,
                    "task":        task,
                    "prompt_type": prompt_type,
                })
    return rows


def main():
    all_rows = []
    for lang in LANGUAGES:
        json_path = SCRIPT_DIR / f"prompts/prompts_{lang}.json"
        if not json_path.exists():
            print(f"[WARN] Missing file: {json_path} — skipping")
            continue
        print(f"[INFO] Processing {lang} ...")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        rows = build_rows(lang, data)
        all_rows.extend(rows)
        print(f"       → {len(rows)} rows added")

    print(f"\n[INFO] Total rows: {len(all_rows)}")

    # Step 1: build dataset with plain strings (no audio loading yet)
    features_str = Features({
        "text_prompt":           Value("string"),
        "audio_prompt_female_1": Value("string"),
        "audio_prompt_female_2": Value("string"),
        "audio_prompt_male_1":   Value("string"),
        "audio_prompt_male_2":   Value("string"),
        "language":              Value("string"),
        "task":                  Value("string"),
        "prompt_type":           Value("string"),
    })

    dataset_dict = {col: [r[col] for r in all_rows] for col in features_str}
    dataset = Dataset.from_dict(dataset_dict, features=features_str)

    # Step 2: cast to Audio lazily — encoding happens during push, not upfront
    features_audio = Features({
        "text_prompt":           Value("string"),
        "audio_prompt_female_1": Audio(sampling_rate=None),
        "audio_prompt_female_2": Audio(sampling_rate=None),
        "audio_prompt_male_1":   Audio(sampling_rate=None),
        "audio_prompt_male_2":   Audio(sampling_rate=None),
        "language":              Value("string"),
        "task":                  Value("string"),
        "prompt_type":           Value("string"),
    })
    dataset = dataset.cast(features_audio)

    print(f"\n[INFO] Dataset:\n{dataset}")

    print(f"\n[INFO] Pushing to HuggingFace: {HF_REPO} ...")
    dataset.push_to_hub(
        HF_REPO,
        split="test",
        token=True,
        max_shard_size="50MB",
    )
    print("[INFO] Done! ✓")


if __name__ == "__main__":
    main()