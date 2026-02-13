def score_tsum(predictions, reference, eval_model=None, lang=None):
    import torch
    from bert_score import score as bert_score
    
    is_batch = isinstance(predictions, list)
    if not is_batch: 
        predictions = [predictions]
    
    # Create references list (same reference repeated for each prediction)
    references = [reference] * len(predictions)
    
    def compute_with_fallback(preds, refs, batch_size=None):
        """Compute BERTScore with automatic batch size reduction on OOM."""
        # Set initial batch_size to total number of predictions if not specified
        if batch_size is None:
            batch_size = len(preds)
        
        try:
            P, R, F1 = bert_score(
                preds, 
                refs, 
                lang=lang, 
                model_type="microsoft/deberta-xlarge-mnli", 
                verbose=False,
                batch_size=batch_size
            )
            return P, R, F1
        except torch.cuda.OutOfMemoryError:
            # Clear cache
            torch.cuda.empty_cache()
            
            # If batch_size is already 1, we can't reduce further
            if batch_size == 1:
                raise RuntimeError("OOM error even with batch_size=1. Consider using a smaller model or CPU.")
            
            # Reduce batch size by half and try again
            new_batch_size = max(1, batch_size // 2)
            print(f"OOM error with batch_size={batch_size}. Retrying with batch_size={new_batch_size}...")
            
            return compute_with_fallback(preds, refs, new_batch_size)
    
    P, R, F1 = compute_with_fallback(predictions, references)
    
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