import json

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


def load_qwen():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    return model, processor

def load_prompt(task, language):
    prompts_path = f"prompts/prompts_{language}.json"
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        if not task.lower() in prompts.keys():
            raise KeyError(f"Task {task} does not have prompts in {prompts_path}.")
    return prompts[task.lower()]


LANGUAGES = [ "de", "cs", "en", "es", "fr", "hu", "it", "nl", "pt", "ru", "sv"]
TASKS = ["ACHAP", "ASR", "MT", "S2ST", "SQA", "SSUM", "ST", "TSUM", "TTS"]


def transcribe_and_evaluate(asr_model="whisper", whisper_model_size="large"):
    if asr_model == "whisper":
        import whisper
        model = whisper.load_model(whisper_model_size)
    elif asr_model == "qwen":
        from qwen_omni_utils import process_mm_info
        model, processor = load_qwen()
    else:
        raise NotImplemntedError
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

                        if asr_model == "whisper":
                            transcription = model.transcribe(audio_path)["text"].strip()
                        elif asr_model == "qwen":
                            conversation = [
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                                    ],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Please transcribe the given audio. Do not add any explanations, follow-up questions or introductions."},
                                        {"type": "audio", "audio": audio_path},
                                    ],
                                },
                            ]

                            # set use audio in video
                            USE_AUDIO_IN_VIDEO = False

                            # Preparation for inference
                            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                            inputs = inputs.to(model.device).to(model.dtype)

                            # Inference: Generation of the output text and audio
                            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                            text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            transcription = text[-1].split("\nassistant")[-1].strip()

                        else:
                            raise NotImplemntedError

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
    MODEL="qwen"

    results = transcribe_and_evaluate(asr_model=MODEL, whisper_model_size="large")
    summarize_results(results)

    with open(f"wer_results_{MODEL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to wer_results_new.json")