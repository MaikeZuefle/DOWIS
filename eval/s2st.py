def score_s2st(predictions, source, eval_model=None, lang=None):
    import numpy as np
    import soundfile as sf
    import torch
    is_batch = isinstance(predictions, list)
    if not is_batch: predictions = [predictions]

    asr_model = eval_model[0]
    comet_model = eval_model[1]
    utmos_model = eval_model[2]

    # get ASR-COMET
    transcripts = asr_model(predictions, batch_size=len(predictions))
    transcripts = [t["text"].strip() for t in transcripts]

    data = [
        {
            "src": source,
            "mt": t
        }
        for t in transcripts
    ]
    
    comet_model_output = comet_model.predict(data, batch_size=len(predictions), gpus=1, progress_bar=False)
    asr_comet_scores = [round(score * 100, 2) for score in comet_model_output.scores]

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
        asr_comet_scores = asr_comet_scores[0]
    metrics = {
        "UTMOS": utmos_scores,
        "ASR-COMET": asr_comet_scores,
    }
    return metrics


