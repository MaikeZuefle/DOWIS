def score_ssum(predictions, reference, eval_model=None, lang=None):
    from bert_score import score as bert_score
    is_batch = isinstance(predictions, list)
    if not is_batch: predictions = [predictions]
    
    # Create references list (same reference repeated for each prediction)
    references = [reference] * len(predictions)
    
    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang=lang, 
        model_type="microsoft/deberta-xlarge-mnli", 
        verbose=False
    )
    
    precision_scores = [round(p.item() * 100, 2) for p in P]
    recall_scores = [round(r.item() * 100, 2) for r in R]
    f1_scores = [round(f.item() * 100, 2) for f in F1]
    

    if not is_batch:
        precision_scores = precision_scores[0]
        recall_scores = recall_scores[0]
        f1_scores = f1_scores[0]
    
    metrics = {
        "BERTScore_P": precision_scores,
        "BERTScore_R": recall_scores,
        "BERTScore_F1": f1_scores,
    }
    
    return metrics