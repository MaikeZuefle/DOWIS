from datasets import load_dataset

def load_mt(language):

    if language == "alb":
        raise ValueError("Albanian is not supported in FLEURS.")

    fleurs_mt = load_dataset("google/fleurs", f"en_us", split="test", trust_remote_code=True)

    sources = [entry["raw_transcription"] for entry in fleurs_mt]

    return {"inputs" : sources, "references": sources} # we append sources because we do QE