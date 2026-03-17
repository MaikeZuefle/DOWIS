import logging

_CHUNKSEG_LANG = {"en": "eng", "de": "deu", "it": "ita", "zh": "zho"}


def score_achap(predictions, reference, eval_model=None, lang=None):
    from mutagen import File as MutaFile
    from chunkseg import evaluate
    if isinstance(predictions, str):
        predictions = [predictions]

    if not isinstance(reference, dict):
        logging.warning("score_achap: reference must be a dict — skipping")
        return {"CollarF1": [0.0] * len(predictions), "GC-BS": [0.0] * len(predictions)}

    ref_timestamps = reference.get("timestamps", [])
    video_id = reference.get("video_id")
    audio = reference.get("audio_path")
    duration = reference.get("duration")
    ref_titles = reference.get("ref_titles")

    n = len(predictions)
    zeros = {"CollarF1": [0.0] * n, "GC-BS": [0.0] * n}

    if not video_id or not audio:
        logging.warning("score_achap: missing video_id or audio_path in reference — skipping")
        return zeros

    # ref_titles from JSON are lists of lists; convert to tuples
    if ref_titles:
        ref_titles = [(t, s) for t, s in ref_titles]

    if duration is None:
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
