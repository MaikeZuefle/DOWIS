from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions

normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])


def score_asr(prediction, reference, eval_model=None):
    wer_score = wer(normalizer(reference), normalizer(prediction))
    wer_score = round(wer_score*100, 2)
    
    metrics = {
        "wer": wer_score,
    }
    return metrics