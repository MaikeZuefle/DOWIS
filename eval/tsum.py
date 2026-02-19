def score_tsum(predictions, reference, eval_model=None, lang=None):
    import torch
    from bert_score import score as bert_score
    
    is_batch = isinstance(predictions, list)
    if not is_batch: 
        predictions = [predictions]
    
    # Create references list (same reference repeated for each prediction)
    references = [reference] * len(predictions)
    
    try:
        P, R, F1 = bert_score(
        predictions, 
        references, 
        lang=lang, 
        model_type="microsoft/deberta-xlarge-mnli",
        rescale_with_baseline=True,
        verbose=False,
        batch_size=len(predictions))
    except:
        try:
            P, R, F1 = bert_score(
            predictions, 
            references, 
            lang=lang, 
            model_type="microsoft/deberta-xlarge-mnli", 
            verbose=False,
            batch_size=1)
        except torch.OutOfMemoryError:
            P, R, F1 =  torch.tensor([0]*len(predictions)), torch.tensor([0]*len(predictions)), torch.tensor([0]*len(predictions))
            print("Zero scores, because of hallucination!")

    
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