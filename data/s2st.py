from datasets import load_dataset
import soundfile as sf
import os
from data.utils import FLEURS_LANG_MAP

def load_s2st(language):

    if language == "sq":
        raise ValueError("Albanian is not supported in FLEURS.")

    base_dir = "data_storage/asr"
    os.makedirs(base_dir, exist_ok=True)

    fleurs_s2st = load_dataset("google/fleurs", FLEURS_LANG_MAP[language], split="test", trust_remote_code=True)
    fleurs_en = load_dataset("google/fleurs", f"en_us", split="test", trust_remote_code=True)
    en_dict = {entry["id"]: entry for entry in fleurs_en}

    audio_paths = [];  references = []

    for idx, entry in enumerate(fleurs_s2st):
        fleurs_idx = entry["id"]
        if fleurs_idx not in en_dict:
            continue
        wav_path = os.path.join(
            base_dir, f"fleurs_{language}_{fleurs_idx}.wav"
        )

        if not os.path.exists(wav_path):
            audio_array = entry["audio"]["array"]
            sr = entry["audio"]["sampling_rate"]
            sf.write(wav_path, audio_array, sr)

        en_entry = en_dict[fleurs_idx]["raw_transcription"]
        audio_paths.append(wav_path)
        references.append(en_entry)

    return {"inputs" : audio_paths, "references": references}