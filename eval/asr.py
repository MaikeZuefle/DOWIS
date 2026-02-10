def score_asr(predictions, reference, eval_model=None, lang=None):
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
        
    wer_scores = [round(wer(reference, prediction)*100, 2) for prediction in predictions]
    
    if not is_batch:
        wer_scores = wer_scores[0]
    
    return {"wer": wer_scores}

