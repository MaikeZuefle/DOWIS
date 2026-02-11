def score_tts(predictions, reference, eval_model=None, lang=None):
    import numpy as np
    import soundfile as sf
    import torch
    from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions

    normalizer = Compose([
        ToLowerCase(),
        ExpandCommonEnglishContractions(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip()
    ])

    is_batch = isinstance(predictions, list)
    if not is_batch: predictions = [predictions]

    asr_model = eval_model[0]
    utmos_model = eval_model[1]

    # get ASR-WER
    transcripts = asr_model(predictions, batch_size=len(predictions))
    transcripts = [t["text"].strip() for t in transcripts]
    asr_wer_scores = [round(wer(reference, t)*100, 2) for t in transcripts]


    # calc UTMOS 
    utmos_scores = []
    for pred in predictions:
        audio_np, sr = sf.read(pred)
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to("cuda")
        utmos_score = utmos_model(audio_tensor, sr).item()
        utmos_score = round(utmos_score, 2)
        utmos_scores.append(utmos_score)

    if not is_batch: 
        utmos_scores = utmos_scores[0]
        asr_wer_scores = asr_wer_scores[0]

    metrics = {
        "UTMOS": utmos_scores,
        "ASR-WER": asr_wer_scores,
    }
    return metrics