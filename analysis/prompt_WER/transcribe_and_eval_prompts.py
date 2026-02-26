import json
import whisper
import jiwer
from tqdm import tqdm
import os

transformation = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def load_prompt(task, language):
    prompts_path = f"prompts/prompts_{language}.json"
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        if not task.lower() in prompts.keys():
            raise KeyError(f"Task {task} does not have prompts in {prompts_path}.")
    return prompts[task.lower()]


LANGUAGES = ["cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "ru", "sv"]
TASKS = ["ACHAP", "ASR", "MT", "S2ST", "SQA", "SSUM", "ST", "TSUM", "TTS"]


def transcribe_and_evaluate(whisper_model_size="large"):
    model = whisper.load_model(whisper_model_size)
    results = {}

    lang_pbar = tqdm(LANGUAGES, desc="Languages", position=0)
    for language in lang_pbar:
        lang_pbar.set_description(f"Language: {language}")
        results[language] = {}

        total_language_samples = 0

        task_pbar = tqdm(TASKS, desc="Tasks", position=1, leave=False)
        for task in task_pbar:
            task_pbar.set_description(f"[{language}] Task: {task}")

            try:
                prompt_dict = load_prompt(task, language)
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

            results[language][task] = {}
            task_samples = 0

            for prompt_type, prompts in prompt_dict.items():
                results[language][task][prompt_type] = {}

                for prompt_idx, p in enumerate(prompts):
                    reference_text = p.get("text", "")
                    audio_entries = []

                    # Collect female recordings
                    if p.get("female_rec"):
                        for rec_idx, audio_path in enumerate(p["female_rec"]):
                            audio_entries.append(("female", audio_path, rec_idx+1))

                    # Collect male recordings
                    if p.get("male_rec"):
                        for rec_idx, audio_path in enumerate(p["male_rec"]):
                            audio_entries.append(("male", audio_path, rec_idx+1))

                    if len(audio_entries) == 0:
                        print(f"[WARNING] No recordings for {language}-{task}-{prompt_type}-prompt{prompt_idx+1}")

                    # Insert recordings into results
                    for gender, audio_path, rec_idx in audio_entries:
                        key = f"prompt_{prompt_idx+1}_{gender}{rec_idx}"
                        transcription = model.transcribe(audio_path)["text"].strip()

                        wer_score = jiwer.wer(reference_text, transcription,
                                              reference_transform=transformation,
                                              hypothesis_transform=transformation)

                        results[language][task][prompt_type][key] = {
                            "audio_path": audio_path,
                            "reference": reference_text,
                            "hypothesis": transcription,
                            "wer": wer_score,
                        }

                        task_samples += 1

            total_language_samples += task_samples

        print(f"[INFO] {language} → Total samples: {total_language_samples}")

    return results


def summarize_results(results):
    print("\n=== WER Summary per Language ===")
    for language, tasks in results.items():
        all_wers = []
        for task, prompt_types in tasks.items():
            for prompt_type, entries in prompt_types.items():
                for key, data in entries.items():
                    all_wers.append(data["wer"])
        if all_wers:
            print(f"[{language}] Mean WER: {sum(all_wers)/len(all_wers):.4f} over {len(all_wers)} samples")

    print("\n=== WER Summary per Task ===")
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
            print(f"[{task}] Mean WER: {sum(wers)/len(wers):.4f} over {len(wers)} samples")


if __name__ == "__main__":
    results = transcribe_and_evaluate(whisper_model_size="large")
    summarize_results(results)

    with open("wer_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to wer_results_new.json")