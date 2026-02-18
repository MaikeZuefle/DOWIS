import json
from pathlib import Path
from collections import defaultdict
from utils import TASK_LANGUAGES, MODEL_TASK_LANGUAGES, TASK_METRICS

# Define the structure
models = ['qwen_omni', 'phi_multimodal']
tasks = ['ASR', 'MT', 'S2ST', 'SSUM', 'TSUM', 'ST', 'TTS', 'ACHAP', 'SQA']

# Define which languages are supported by each task

# All possible languages (union of all task languages)
all_languages = sorted(list(set(lang for langs in TASK_LANGUAGES.values() for lang in langs)))

# Define which metric(s) to use for each task
# Tasks can have single metric (string) or multiple metrics (list)

# Base directory
base_dir = 'eval_outputs'

# Dictionary to store raw results
# Structure: raw_results[model][task][metric][modality] = list of {language, value}
raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

# Track missing data
missing_files = defaultdict(lambda: defaultdict(list))
missing_modalities = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

# Process each model and task
print("Processing files...")
files_processed = 0
files_missing = 0

print_task_order = ['ASR', 'SQA', 'ACHAP', 'TTS', 'MT', 'ST', 'TSUM', 'SSUM', 'S2ST']
print_model_order = ['phi_multimodal', 'qwen_omni']

for task in print_task_order:
    for model in print_model_order:
        task_metrics = TASK_METRICS.get(task, 'wer')
        
        # Handle both single metric (string) and multiple metrics (list)
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]
        
        # Get supported languages for this task
        task_langs = MODEL_TASK_LANGUAGES.get(model, {}).get(task, TASK_LANGUAGES.get(task, []))
        
        for lang in task_langs:
            file_path = Path(base_dir) / model / task / f'{lang}_eval.json'
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    files_processed += 1
                    
                    # Extract metric for each modality
                    if 'results' in data and 'per_modality' in data['results']:
                        modalities_data = data['results']['per_modality']
                        
                        # Process each metric for this task
                        for metric in task_metrics:
                            # Text prompt
                            if 'text_prompt' in modalities_data and metric in modalities_data['text_prompt']:
                                metric_value = modalities_data['text_prompt'][metric]['mean']
                                raw_results[model][task][metric]['text_prompt'].append({
                                    'language': lang,
                                    'value': metric_value
                                })
                            else:
                                missing_modalities[model][task][metric]['text_prompt'].append(lang)
                            
                            # Female audio prompt
                            if 'f_audio_prompt' in modalities_data and metric in modalities_data['f_audio_prompt']:
                                metric_value = modalities_data['f_audio_prompt'][metric]['mean']
                                raw_results[model][task][metric]['f_audio_prompt'].append({
                                    'language': lang,
                                    'value': metric_value
                                })
                            else:
                                missing_modalities[model][task][metric]['f_audio_prompt'].append(lang)
                            
                            # Male audio prompt
                            if 'm_audio_prompt' in modalities_data and metric in modalities_data['m_audio_prompt']:
                                metric_value = modalities_data['m_audio_prompt'][metric]['mean']
                                raw_results[model][task][metric]['m_audio_prompt'].append({
                                    'language': lang,
                                    'value': metric_value
                                })
                            else:
                                missing_modalities[model][task][metric]['m_audio_prompt'].append(lang)
                    else:
                        # File exists but no per_modality data
                        for metric in task_metrics:
                            missing_modalities[model][task][metric]['text_prompt'].append(lang)
                            missing_modalities[model][task][metric]['f_audio_prompt'].append(lang)
                            missing_modalities[model][task][metric]['m_audio_prompt'].append(lang)
                    
                    metrics_str = ', '.join(task_metrics)
                    print(f"✓ {model}/{task}/{lang} (metrics: {metrics_str})")
                except Exception as e:
                    print(f"✗ {model}/{task}/{lang}: Error - {str(e)}")
                    files_missing += 1
                    missing_files[model][task].append(lang)
            else:
                files_missing += 1
                missing_files[model][task].append(lang)
                print(f"⊗ {model}/{task}/{lang}: File not found")

print(f"\n{'='*80}")
print(f"Files processed: {files_processed}")
print(f"Files missing: {files_missing}")
total_expected = sum(len(TASK_LANGUAGES.get(task, [])) for task in tasks) * len(models)
print(f"Total expected: {total_expected}")
print(f"{'='*80}\n")

print("Calculating averages...")

# Calculate averages
output_results = {}

for model in models:
    output_results[model] = {}
    
    for task in tasks:
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]
        
        output_results[model][task] = {
            'metrics': task_metrics
        }
        
        # Process each metric for this task
        for metric in task_metrics:
            output_results[model][task][metric] = {}
            
            # Text prompt average
            if 'text_prompt' in raw_results[model][task][metric] and len(raw_results[model][task][metric]['text_prompt']) > 0:
                values = [item['value'] for item in raw_results[model][task][metric]['text_prompt']]
                output_results[model][task][metric]['text_prompt'] = {
                    'average': sum(values) / len(values),
                    'num_languages': len(values),
                    'languages': sorted([item['language'] for item in raw_results[model][task][metric]['text_prompt']]),
                    'per_language': {item['language']: item['value'] for item in raw_results[model][task][metric]['text_prompt']}
                }
            else:
                output_results[model][task][metric]['text_prompt'] = None
            
            # Female audio prompt average
            if 'f_audio_prompt' in raw_results[model][task][metric] and len(raw_results[model][task][metric]['f_audio_prompt']) > 0:
                values = [item['value'] for item in raw_results[model][task][metric]['f_audio_prompt']]
                output_results[model][task][metric]['f_audio_prompt'] = {
                    'average': sum(values) / len(values),
                    'num_languages': len(values),
                    'languages': sorted([item['language'] for item in raw_results[model][task][metric]['f_audio_prompt']]),
                    'per_language': {item['language']: item['value'] for item in raw_results[model][task][metric]['f_audio_prompt']}
                }
            else:
                output_results[model][task][metric]['f_audio_prompt'] = None
            
            # Male audio prompt average
            if 'm_audio_prompt' in raw_results[model][task][metric] and len(raw_results[model][task][metric]['m_audio_prompt']) > 0:
                values = [item['value'] for item in raw_results[model][task][metric]['m_audio_prompt']]
                output_results[model][task][metric]['m_audio_prompt'] = {
                    'average': sum(values) / len(values),
                    'num_languages': len(values),
                    'languages': sorted([item['language'] for item in raw_results[model][task][metric]['m_audio_prompt']]),
                    'per_language': {item['language']: item['value'] for item in raw_results[model][task][metric]['m_audio_prompt']}
                }
            else:
                output_results[model][task][metric]['m_audio_prompt'] = None
            
            # Combined audio (both male and female)
            all_audio_values = []
            all_audio_langs = set()
            
            if 'f_audio_prompt' in raw_results[model][task][metric]:
                for item in raw_results[model][task][metric]['f_audio_prompt']:
                    all_audio_values.append(item['value'])
                    all_audio_langs.add(item['language'])
            
            if 'm_audio_prompt' in raw_results[model][task][metric]:
                for item in raw_results[model][task][metric]['m_audio_prompt']:
                    all_audio_values.append(item['value'])
                    all_audio_langs.add(item['language'])
            
            if len(all_audio_values) > 0:
                output_results[model][task][metric]['audio_prompt_combined'] = {
                    'average': sum(all_audio_values) / len(all_audio_values),
                    'num_values': len(all_audio_values),
                    'num_languages': len(all_audio_langs),
                    'languages': sorted(list(all_audio_langs)),
                    'note': 'Average over both male and female audio prompts'
                }
            else:
                output_results[model][task][metric]['audio_prompt_combined'] = None

# Create metadata about missing data
total_expected = sum(len(TASK_LANGUAGES.get(task, [])) for task in tasks) * len(models)

metadata = {
    'files_processed': files_processed,
    'files_missing': files_missing,
    'total_expected': total_expected,
    'task_metrics': TASK_METRICS,
    'task_languages': TASK_LANGUAGES,
    'missing_files_by_model_task': {
        model: {
            task: langs for task, langs in tasks_dict.items() if langs
        } for model, tasks_dict in missing_files.items()
    },
    'missing_modalities_by_model_task': dict(missing_modalities)
}

# Combine results and metadata
full_output = {
    'metadata': metadata,
    'results': output_results
}

# Save to JSON
output_file = 'modality_averages.json'
with open(output_file, 'w') as f:
    json.dump(full_output, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Print summary table
print("\n" + "="*150)
print("SUMMARY: Average Scores by Modality (across available languages)")
print("="*150)
print(f"{'Task':<10} {'Metric':<18} {'Text':>12} {'Combined':>12} {'Male':>12} {'Female':>12} {'Langs':>8} {'Model':<15}")
print("-"*150)

for task in print_task_order:
    for model in print_model_order:
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]
        
        # Get expected language count for this task
        expected_langs = len(MODEL_TASK_LANGUAGES.get(model, {}).get(task, TASK_LANGUAGES.get(task, [])))
        if expected_langs == 0:
            continue
        
        for metric in task_metrics:
            text_val = "N/A"
            f_val = "N/A"
            m_val = "N/A"
            combined_val = "N/A"
            num_langs = 0
            
            if output_results[model][task][metric]['text_prompt']:
                text_val = f"{output_results[model][task][metric]['text_prompt']['average']:.2f}"
                num_langs = output_results[model][task][metric]['text_prompt']['num_languages']
            
            if output_results[model][task][metric]['f_audio_prompt']:
                f_val = f"{output_results[model][task][metric]['f_audio_prompt']['average']:.2f}"
                if num_langs == 0:
                    num_langs = output_results[model][task][metric]['f_audio_prompt']['num_languages']
            
            if output_results[model][task][metric]['m_audio_prompt']:
                m_val = f"{output_results[model][task][metric]['m_audio_prompt']['average']:.2f}"
                if num_langs == 0:
                    num_langs = output_results[model][task][metric]['m_audio_prompt']['num_languages']
            
            if output_results[model][task][metric]['audio_prompt_combined']:
                combined_val = f"{output_results[model][task][metric]['audio_prompt_combined']['average']:.2f}"
                if num_langs == 0:
                    num_langs = output_results[model][task][metric]['audio_prompt_combined']['num_languages']
            
            lang_str = f"{num_langs}/{expected_langs}" if expected_langs > 0 else "0/0"
            
            print(f"{task:<10} {metric:<18} {text_val:>12} {combined_val:>12} {m_val:>12} {f_val:>12} {lang_str:>8} {model:<15}")

print("="*150)
print("\nNote: 'Combined' is the average over both male and female audio prompts")
print("      'Langs' shows how many languages have data / expected languages for that task")
print("      For WER: lower is better. For CometQE, BERTScore, and UTMOS: higher is better")
print("\nLanguage support by task:")
for task in tasks:
    langs = TASK_LANGUAGES.get(task, [])
    print(f"  {task}: {len(langs)} languages ({', '.join(langs)})")
print("\nTasks with multiple metrics:")
print("  - TTS: UTMOS (speech quality), ASR_WER (intelligibility)")
print("  - S2ST: UTMOS (speech quality), ASR_CometQE (translation quality)")
print("\nMissing data is handled as follows:")
print("  - Missing files are skipped")
print("  - Averages are calculated only over available languages")
print("  - If no data is available for a task/metric, it shows as 'N/A'")
print("  - Check 'metadata' section in JSON for details on missing data")