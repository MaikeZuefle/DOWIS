import json


def summarize_results(results):
    summary = {
        "per_language": {},
        "per_task": {}
    }

    # === WER Summary per Language ===
    for language, tasks in results.items():
        all_wers = []
        for task, prompt_types in tasks.items():
            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    all_wers.append(data["wer"])

        if all_wers:
            summary["per_language"][language] = {
                "mean_wer": sum(all_wers) / len(all_wers),
                "num_samples": len(all_wers)
            }

    # === WER Summary per Task ===
    task_wers = {}

    for language, tasks in results.items():
        for task, prompt_types in tasks.items():
            if task not in task_wers:
                task_wers[task] = []

            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    task_wers[task].append(data["wer"])

    for task, wers in task_wers.items():
        if wers:
            summary["per_task"][task] = {
                "mean_wer": sum(wers) / len(wers),
                "num_samples": len(wers)
            }

    return summary


if __name__ == "__main__":
    # Load previous results
    with open("wer_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = summarize_results(results)

    # Save summary
    with open("wer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved to wer_summary.json")

    # Optional: also print nicely
    print("\n=== WER Summary per Language ===")
    for lang, data in summary["per_language"].items():
        print(f"[{lang}] Mean WER: {data['mean_wer']:.4f} over {data['num_samples']} samples")

    print("\n=== WER Summary per Task ===")
    for task, data in summary["per_task"].items():
        print(f"[{task}] Mean WER: {data['mean_wer']:.4f} over {data['num_samples']} samples")

