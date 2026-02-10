def score_st(predictions, reference, eval_model=None, lang=None):

    is_batch = isinstance(predictions, list)
    if not is_batch: predictions = [predictions]
    
    data = [
        {
            "src": reference,
            "mt": prediction
        }
        for prediction in predictions
    ]
    
    model_output = eval_model.predict(data, batch_size=len(predictions), gpus=1, progress_bar=False)
    scores = [round(score * 100, 2) for score in model_output.scores]
    
    if not is_batch: scores = scores[0]
    
    return {"CometQE": scores}