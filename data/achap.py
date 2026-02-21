from datasets import load_dataset
import soundfile as sf
import os

def load_achap(language):

    if language != "en":
        raise ValueError(f"{language} is not supported. Only English (en) is available.")

    base_dir = "data_storage/achap"
    os.makedirs(base_dir, exist_ok=True)

    ytseg = load_dataset("retkowski/ytseg", "text", split="test", trust_remote_code=True, download_mode="force_redownload")

    audio_paths = []
    references = []
    chapter_titles = []
    video_ids = []

    for idx, entry in enumerate(ytseg):
        if entry["duration"] > 1200:
            continue

        channel_id = entry["channel_id"]
        video_id = entry["video_id"]
        wav_path = os.path.join(base_dir, channel_id, f"{video_id}_.mp3")

        if not os.path.exists(wav_path):
            audio_array = entry["audio"]["array"]
            sr = entry["audio"]["sampling_rate"]
            sf.write(wav_path, audio_array, sr)

        audio_paths.append(wav_path)
        references.append(entry["chapter_timestamps"])
        chapter_titles.append(entry["chapter_titles"])
        video_ids.append(video_id)

    return {"inputs" : audio_paths, "references": references, "chapter_titles": chapter_titles, "video_ids": video_ids}