import json

def get_gender(key: str):
    """Extract gender from entry key, e.g. 'prompt_1_female1', 'prompt_2_male1'."""
    key = key.lower()
    if "female" in key:
        return "female"
    if "male" in key:
        return "male"
    return None

def summarize_results(results):
    summary = {
        "per_language": {},
        "per_task": {},
        "per_language_per_task": {},
        "per_gender": {},
        "per_language_per_gender": {},
        "per_language_per_task_per_gender": {}
    }

    # === Per language / per language+task / per language+task+gender ===
    for language, tasks in results.items():
        all_wers = []
        summary["per_language_per_task"][language] = {}
        summary["per_language_per_gender"][language] = {}
        summary["per_language_per_task_per_gender"][language] = {}

        lang_gender_wers = {}

        for task, prompt_types in tasks.items():
            task_wers = []
            task_gender_wers = {}
            summary["per_language_per_task_per_gender"][language][task] = {}

            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    gender = get_gender(key)
                    wer = data["wer"]
                    all_wers.append(wer)
                    task_wers.append(wer)
                    if gender:
                        task_gender_wers.setdefault(gender, []).append(wer)
                        lang_gender_wers.setdefault(gender, []).append(wer)

            # Per language per task
            if task_wers:
                summary["per_language_per_task"][language][task] = {
                    "mean_wer": sum(task_wers) / len(task_wers),
                    "num_samples": len(task_wers)
                }

            # Per language per task per gender
            for gender, wers in task_gender_wers.items():
                summary["per_language_per_task_per_gender"][language][task][gender] = {
                    "mean_wer": sum(wers) / len(wers),
                    "num_samples": len(wers)
                }

            # Gender gap for this language+task
            tg = summary["per_language_per_task_per_gender"][language][task]
            if "male" in tg and "female" in tg:
                tg["gender_gap"] = round(tg["female"]["mean_wer"] - tg["male"]["mean_wer"], 6)

        # Per language
        if all_wers:
            summary["per_language"][language] = {
                "mean_wer": sum(all_wers) / len(all_wers),
                "num_samples": len(all_wers)
            }

        # Per language per gender
        for gender, wers in lang_gender_wers.items():
            summary["per_language_per_gender"][language][gender] = {
                "mean_wer": sum(wers) / len(wers),
                "num_samples": len(wers)
            }

        # Gender gap per language
        lg = summary["per_language_per_gender"][language]
        if "male" in lg and "female" in lg:
            lg["gender_gap"] = round(lg["female"]["mean_wer"] - lg["male"]["mean_wer"], 6)

    # === Per task (across languages) ===
    task_wers = {}
    for language, tasks in results.items():
        for task, prompt_types in tasks.items():
            task_wers.setdefault(task, [])
            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    task_wers[task].append(data["wer"])

    for task, wers in task_wers.items():
        if wers:
            summary["per_task"][task] = {
                "mean_wer": sum(wers) / len(wers),
                "num_samples": len(wers)
            }

    # === Per gender (global) ===
    global_gender_wers = {}
    for language, tasks in results.items():
        for task, prompt_types in tasks.items():
            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    gender = get_gender(key)
                    if gender:
                        global_gender_wers.setdefault(gender, []).append(data["wer"])

    for gender, wers in global_gender_wers.items():
        summary["per_gender"][gender] = {
            "mean_wer": sum(wers) / len(wers),
            "num_samples": len(wers)
        }
    if "male" in summary["per_gender"] and "female" in summary["per_gender"]:
        summary["per_gender"]["gender_gap"] = round(
            summary["per_gender"]["female"]["mean_wer"] - summary["per_gender"]["male"]["mean_wer"], 6
        )

    return summary


if __name__ == "__main__":
    with open("wer_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = summarize_results(results)

    with open("wer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to wer_summary.json")

    print("\n=== WER Summary per Language ===")
    for lang, data in summary["per_language"].items():
        print(f"[{lang}] Mean WER: {data['mean_wer']:.4f} over {data['num_samples']} samples")

    print("\n=== WER Summary per Task ===")
    for task, data in summary["per_task"].items():
        print(f"[{task}] Mean WER: {data['mean_wer']:.4f} over {data['num_samples']} samples")

    print("\n=== WER Summary per Gender (Global) ===")
    for gender in ("male", "female"):
        if gender in summary["per_gender"]:
            d = summary["per_gender"][gender]
            print(f"[{gender}] Mean WER: {d['mean_wer']:.4f} over {d['num_samples']} samples")
    if "gender_gap" in summary["per_gender"]:
        gap = summary["per_gender"]["gender_gap"]
        print(f"Gender gap (female - male): {gap:+.4f}")

    print("\n=== WER per Language per Gender ===")
    for lang, genders in summary["per_language_per_gender"].items():
        parts = []
        for gender in ("male", "female"):
            if gender in genders:
                d = genders[gender]
                parts.append(f"{gender}: {d['mean_wer']:.4f} (n={d['num_samples']})")
        if parts:
            gap_str = f"  gap: {genders['gender_gap']:+.4f}" if "gender_gap" in genders else ""
            print(f"[{lang}] {' | '.join(parts)}{gap_str}")

    print("\n=== WER per Language per Task per Gender ===")
    for lang, tasks in summary["per_language_per_task_per_gender"].items():
        for task, genders in tasks.items():
            parts = []
            for gender in ("male", "female"):
                if gender in genders:
                    d = genders[gender]
                    parts.append(f"{gender}: {d['mean_wer']:.4f} (n={d['num_samples']})")
            if parts:
                gap_str = f"  gap: {genders['gender_gap']:+.4f}" if "gender_gap" in genders else ""
                print(f"[{lang}][{task}] {' | '.join(parts)}{gap_str}")