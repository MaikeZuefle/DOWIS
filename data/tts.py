import os
from datasets import load_dataset
from data.utils import FLEURS_LANG_MAP

def load_tts(language):

    if language == "sq":
        raise ValueError("Albanian is not supported in FLEURS.")


    fleurs_tts = load_dataset("google/fleurs", FLEURS_LANG_MAP[language], split="test", trust_remote_code=True)

    sources = [entry["raw_transcription"] for entry in fleurs_tts]

    return {"inputs" : sources, "references": sources} 