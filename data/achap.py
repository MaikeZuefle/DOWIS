from datasets import load_dataset
import os

def load_achap(language):

    if language != "en":
        raise ValueError(f"{language} is not supported. Only English (en) is available.")

    base_dir = "data_storage/achap"
    os.makedirs(base_dir, exist_ok=True)

    ytseg = load_dataset("retkowski/ytseg", "text", split="test")

    audio_paths = []
    references = []

    for entry in ytseg:
        if entry["duration"] > 1200:
            continue

        channel_id = entry["channel_id"]
        video_id = entry["video_id"]
        audio_path = os.path.join(base_dir, channel_id, f"{video_id}_.mp3")

        raw_chapters = entry.get("raw_chapters") or []
        ref_titles = [(c["title"], c["start_time"]) for c in raw_chapters]

        audio_paths.append(audio_path)
        references.append({
            "timestamps": entry["chapter_timestamps"],
            "video_id": video_id,
            "audio_path": audio_path,
            "duration": entry["duration"],
            "ref_titles": ref_titles,
        })

    return {"inputs": audio_paths, "references": references}
