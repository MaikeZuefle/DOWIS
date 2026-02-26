import json
import logging
import os
from chunkseg import evaluate
from mutagen import File as MutaFile


_DATA_DIR = ""
_YTSEG_INDEX = ""
_CHUNKSEG_LANG = {"en": "eng", "de": "deu", "it": "ita", "zh": "zho"}


def _build_index() -> dict:
    idx = {}
    with open(_YTSEG_INDEX) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["video_id"] not in idx:
                idx[d["video_id"]] = d["channel_id"]
    return idx


_VIDEO_INDEX = None


def _get_channel_id(video_id: str) -> str:
    global _VIDEO_INDEX
    if _VIDEO_INDEX is None:
        _VIDEO_INDEX = _build_index()
    channel_id = _VIDEO_INDEX.get(video_id)
    if channel_id is None:
        raise KeyError(f"video_id={video_id!r} not found in YTSeg index")
    return channel_id


def _resolve_audio(video_id: str) -> str:
    channel_id = _get_channel_id(video_id)
    path = os.path.join(_DATA_DIR, channel_id, f"{video_id}.mp3")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No audio found for video_id={video_id!r}")


def _load_ref_titles(video_id: str) -> list:
    channel_id = _get_channel_id(video_id)
    path = os.path.join(_DATA_DIR, channel_id, f"{video_id}.chapters.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        chapters = json.load(f)
    return [(c["title"], c["start_time"]) for c in chapters]


def score_achap(predictions, reference, eval_model=None, lang=None):
    if isinstance(predictions, str):
        predictions = [predictions]

    if isinstance(reference, dict):
        ref_timestamps = reference.get("timestamps", [])
        video_id = reference.get("video_id")
    else:
        ref_timestamps = list(reference)
        video_id = None

    n = len(predictions)
    zeros = {"CollarF1": [0.0] * n, "GC-BS": [0.0] * n}

    if not video_id:
        logging.warning("score_achap: no video_id in reference — skipping")
        return zeros

    audio = _resolve_audio(video_id)

    ref_titles = _load_ref_titles(video_id)
    duration = MutaFile(audio).info.length
    chunkseg_lang = _CHUNKSEG_LANG.get(lang, "eng")

    collar_f1s, gc_bs_f1s = [], []
    for hyp in predictions:
        try:
            result = evaluate(
                hypothesis=hyp,
                reference=ref_timestamps,
                duration=duration,
                audio=audio,
                format="markdown",
                lang=chunkseg_lang,
                reference_titles=ref_titles if ref_titles else None,
                collar=3.0,
                tolerance=5.0,
            )
            collar_f1s.append(result.get("collar_f1", 0.0))
            gc_bs_f1s.append(result.get("gc_bs_f1", 0.0))
        except Exception as e:
            logging.warning(f"score_achap: evaluate() failed for {video_id!r}: {e}")
            collar_f1s.append(0.0)
            gc_bs_f1s.append(0.0)

    return {"CollarF1": collar_f1s, "GC-BS": gc_bs_f1s}
