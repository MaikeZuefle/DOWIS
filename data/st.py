import os
from datasets import load_dataset
import soundfile as sf

def load_st(language):

    if language == "sq":
        raise ValueError("Albanian is not supported in FLEURS.")

    base_dir = "data_storage/st"
    os.makedirs(base_dir, exist_ok=True)

    fleurs_st = load_dataset("google/fleurs", f"en_us", split="test", trust_remote_code=True)

    audio_paths = [];  sources = []

    for idx, entry in enumerate(fleurs_st):
        wav_path = os.path.join(
            base_dir, f"fleurs_en_us_{idx}.wav"
        )

        if not os.path.exists(wav_path):
            audio_array = entry["audio"]["array"]
            sr = entry["audio"]["sampling_rate"]
            sf.write(wav_path, audio_array, sr)

        audio_paths.append(wav_path)
        sources.append(entry["raw_transcription"])

    return {"inputs" : audio_paths, "references": sources} # we append sources because we do QE