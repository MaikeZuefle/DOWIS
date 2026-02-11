import argparse
import logging
import os
import json
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm


# utils
from utils import set_up_logging, TASK_MODALITY_MAPPER
from eval.utils import get_score_function


class EvaluationResults:
    """Class to accumulate and compute evaluation results."""
    
    def __init__(self):
        # Store all individual scores
        # Structure: {prompt_type: {prompt_modality: {metric_name: [scores]}}}
        self.scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Track counts for each category
        self.counts = defaultdict(lambda: defaultdict(int))

    
    def add_score(self, prompt_type: str, prompt_modality: str, metrics: Dict[str, float]):
        """Add a single evaluation result."""
        for metric_name, score in metrics.items():
            self.scores[prompt_type][prompt_modality][metric_name].append(score)
        self.counts[prompt_type][prompt_modality] += 1
    
    def compute_averages(self) -> Dict[str, Any]:
        """Compute all relevant averages."""
        results = {
            "per_prompt_type": {},
            "per_modality": {},
            "overall": {}
        }
        
        # 1. Average per prompt type and modality (most granular)
        for prompt_type, modalities in self.scores.items():
            results["per_prompt_type"][prompt_type] = {}
            for prompt_modality, metrics in modalities.items():
                results["per_prompt_type"][prompt_type][prompt_modality] = {}
                for metric_name, scores in metrics.items():
                    avg = sum(scores) / len(scores) if scores else 0.0
                    results["per_prompt_type"][prompt_type][prompt_modality][metric_name] = {
                        "mean": avg,
                        "count": len(scores)
                    }
        
        # 2. Average per prompt type (across all modalities)
        for prompt_type, modalities in self.scores.items():
            all_scores_for_type = defaultdict(list)
            for prompt_modality, metrics in modalities.items():
                for metric_name, scores in metrics.items():
                    all_scores_for_type[metric_name].extend(scores)
            
            results["per_prompt_type"][prompt_type]["all_prompt_modalities"] = {}
            for metric_name, scores in all_scores_for_type.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                results["per_prompt_type"][prompt_type]["all_prompt_modalities"][metric_name] = {
                    "mean": avg,
                    "count": len(scores)
                }
        
        # 3. Average per modality (across all prompt types)
        all_modalities = set()
        for modalities in self.scores.values():
            all_modalities.update(modalities.keys())
        
        for modality in all_modalities:
            all_scores_for_modality = defaultdict(list)
            for prompt_type, modalities in self.scores.items():
                if modality in modalities:
                    for metric_name, scores in modalities[modality].items():
                        all_scores_for_modality[metric_name].extend(scores)
            
            results["per_modality"][modality] = {}
            for metric_name, scores in all_scores_for_modality.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                results["per_modality"][modality][metric_name] = {
                    "mean": avg,
                    "count": len(scores)
                }
        
        # 4. Overall average (across everything)
        all_scores_overall = defaultdict(list)
        for prompt_type, modalities in self.scores.items():
            for prompt_modality, metrics in modalities.items():
                for metric_name, scores in metrics.items():
                    all_scores_overall[metric_name].extend(scores)
        
        for metric_name, scores in all_scores_overall.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            results["overall"][metric_name] = {
                "mean": avg,
                "count": len(scores)
            }
        
        return results
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the evaluation."""
        # Total evaluations (each sample evaluated with multiple prompt types & modalities)
        total_evaluations = sum(sum(counts.values()) for counts in self.counts.values())
        
        # Count evaluations per prompt type
        evaluations_per_prompt_type = {}
        for prompt_type, modalities in self.counts.items():
            evaluations_per_prompt_type[prompt_type] = sum(modalities.values())
        
        # Count evaluations per modality
        evaluations_per_modality = defaultdict(int)
        for prompt_type, modalities in self.counts.items():
            for modality, count in modalities.items():
                evaluations_per_modality[modality] += count
        
        # Infer number of unique samples (total / (num_prompt_types * num_modalities))
        num_prompt_types = len(evaluations_per_prompt_type)
        num_modalities = len(evaluations_per_modality)
        unique_samples = total_evaluations // (num_prompt_types * num_modalities) if num_prompt_types > 0 and num_modalities > 0 else 0
        
        stats = {
            "unique_samples": unique_samples,
            "total_evaluations": total_evaluations,
            "num_prompt_types": num_prompt_types,
            "num_modalities": num_modalities,
            "evaluations_per_prompt_type": evaluations_per_prompt_type,
            "evaluations_per_modality": dict(evaluations_per_modality)
        }
        
        return stats

def main(out_folder, model, task, lang, predictions_folder=None):
    
    # Setting paths
    if predictions_folder is None:
        predictions_folder = out_folder
    
    predictions_file_path = f"{predictions_folder}/{model}/{task}/{lang}.jsonl"
    eval_output_path = f"{out_folder}/{model}/{task}/{lang}_eval.json"
    
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    set_up_logging(eval_output_path.replace("_eval.json", "_eval.log"))
    
    # Get task modality info
    modality = TASK_MODALITY_MAPPER[task]["modality"]
    output_modality = TASK_MODALITY_MAPPER[task]["output_modality"]

    evaluate_prediction, eval_model = get_score_function(task)

    # Logging
    logging.info("=" * 80)
    logging.info("Starting Evaluation")
    logging.info("=" * 80)
    logging.info(f"Task: {task}")
    logging.info(f"Language: {lang}")
    logging.info(f"Model: {model}")
    logging.info(f"Input Modality: {modality}")
    logging.info(f"Output Modality: {output_modality}")
    logging.info(f"Predictions file: {predictions_file_path}")
    logging.info(f"Evaluation output: {eval_output_path}")
    
    # Check if predictions file exists
    if not os.path.exists(predictions_file_path):
        logging.error(f"Predictions file not found: {predictions_file_path}")
        return
    
    # Initialize results accumulator
    eval_results = EvaluationResults()
    
    # Read and evaluate predictions
    logging.info("Reading predictions and computing metrics...")
    
    # First pass: count total lines for progress bar
    total_lines = 0
    with open(predictions_file_path, "r", encoding="utf-8") as f_count:
        for line in f_count:
            if line.strip():
                total_lines += 1
    
    logging.info(f"Found {total_lines} samples to evaluate")
    
    line_count = 0
    error_count = 0
    
    with open(predictions_file_path, "r", encoding="utf-8") as f_in:
        for line_idx, line in tqdm(enumerate(f_in), total=total_lines, desc="Evaluating"):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1

            sample = json.loads(line)
            reference = sample["ref"]
            predicted = sample["predicted"]

            # Collect all predictions for this sample (up to 15: 5 prompt types × 3 modalities)
            batch_predictions = []
            batch_metadata = []  # Store (prompt_type, prompt_modality) for each prediction
            
            # Iterate over all prompt types (basic, formal, informal, detailed, short)

            for prompt_type, predictions_dict in predicted.items():
                # Skip metadata fields
                if prompt_type in ["prompt_number", "spk_number"]:
                    continue

                if task in ["TTS", "S2ST"]:
                    # predictions will be the audio paths
                    a_dir = predictions_file_path.replace(".jsonl", "_wavs")
                    predictions_dict["text_prompt"] = f"{a_dir}/{line_idx}_{prompt_type}_text_prompt.wav"
                    predictions_dict["f_audio_prompt"] = f"{a_dir}/{line_idx}_{prompt_type}_f_audio_prompt.wav"
                    predictions_dict["m_audio_prompt"] = f"{a_dir}/{line_idx}_{prompt_type}_m_audio_prompt.wav"    


                # Collect text prompt (always present)
                if "text_prompt" in predictions_dict:
                    batch_predictions.append(predictions_dict["text_prompt"])
                    batch_metadata.append((prompt_type, "text_prompt"))
                
                # Collect female audio prompt (if present)
                if "f_audio_prompt" in predictions_dict:
                    batch_predictions.append(predictions_dict["f_audio_prompt"])
                    batch_metadata.append((prompt_type, "f_audio_prompt"))
                
                # Collect male audio prompt (if present)
                if "m_audio_prompt" in predictions_dict:
                    batch_predictions.append(predictions_dict["m_audio_prompt"])
                    batch_metadata.append((prompt_type, "m_audio_prompt"))
            
            # Evaluate all predictions for this sample in one batch
            if batch_predictions:
                batch_metrics = evaluate_prediction(
                    batch_predictions, 
                    reference, 
                    eval_model=eval_model,
                    lang=lang,
                )
                
                # Add scores back to results using metadata
                for idx, (prompt_type, prompt_modality) in enumerate(batch_metadata):
                    # Extract metrics for this specific prediction
                    sample_metrics = {
                        metric_name: scores[idx] 
                        for metric_name, scores in batch_metrics.items()
                    }
                    eval_results.add_score(prompt_type, prompt_modality, sample_metrics)
                    
    logging.info(f"Processed {line_count} samples")
    if error_count > 0:
        logging.warning(f"Encountered {error_count} errors during processing")
    
    # Compute all averages
    logging.info("Computing averages...")
    averaged_results = eval_results.compute_averages()
    summary_stats = eval_results.get_summary_stats()
    
    # Prepare final output
    final_output = {
        "task": task,
        "language": lang,
        "model": model,
        "input_modality": modality,
        "output_modality": output_modality,
        "summary_stats": summary_stats,
        "results": averaged_results
    }
    
    # Write results to file
    logging.info(f"Writing evaluation results to {eval_output_path}")
    with open(eval_output_path, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, indent=2, ensure_ascii=False)
    
    # Log summary
    logging.info("=" * 80)
    logging.info("Evaluation Summary")
    logging.info("=" * 80)
    logging.info(f"Unique samples: {summary_stats['unique_samples']}")
    logging.info(f"Total evaluations: {summary_stats['total_evaluations']}")
    logging.info(f"  ({summary_stats['num_prompt_types']} prompt types × {summary_stats['num_modalities']} modalities)")
    logging.info(f"\nEvaluations per prompt type:")
    for prompt_type, count in summary_stats['evaluations_per_prompt_type'].items():
        logging.info(f"  - {prompt_type}: {count}")
    logging.info(f"\nEvaluations per modality:")
    for modality, count in summary_stats['evaluations_per_modality'].items():
        logging.info(f"  - {modality}: {count}")
    
    logging.info("\nOverall results:")
    for metric_name, result in averaged_results["overall"].items():
        logging.info(f"  {metric_name}: {result['mean']:.4f} (n={result['count']})")
    
    logging.info("=" * 80)
    logging.info("Evaluation complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    LANGS = ["cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "ru", "sq", "sv"]
    TASKS = ["ACHAP", "ASR", "MT", "S2ST", "SLU", "SQA", "SSUM", "ST", "TSUM", "TTS"]
    MODELS = ["phi_multimodal", "qwen_omni"]

    parser = argparse.ArgumentParser(description="Evaluate MCIF predictions.")

    parser.add_argument(
        "--lang", choices=LANGS, default=LANGS[0], help="Language to evaluate"
    )
    parser.add_argument(
        "--task", choices=TASKS, default=TASKS[0], help="Task to evaluate"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0], help="Model type")
    parser.add_argument(
        "--out_folder", default="evaluation_results", help="Output folder for evaluation results"
    )
    parser.add_argument(
        "--predictions_folder", default=None, 
        help="Folder containing predictions (defaults to out_folder if not specified)"
    )

    args = parser.parse_args()
    main(
        out_folder=args.out_folder,
        model=args.model,
        task=args.task,
        lang=args.lang,
        predictions_folder=args.predictions_folder,
    )

    # Usage examples:
    # python eval.py --lang es --model phi_multimodal --task S2ST --predictions_folder generated_output --out_folder evaluation_results
    # python eval.py --lang en --model qwen_omni --task ASR