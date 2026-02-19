import json
from pathlib import Path
from collections import defaultdict
from utils import TASK_LANGUAGES, MODEL_TASK_LANGUAGES, TASK_METRICS

# Define the structure
models = ['qwen_omni', 'phi_multimodal']
tasks = ['ASR', 'MT', 'S2ST', 'SSUM', 'TSUM', 'ST', 'TTS', 'ACHAP', 'SQA']

# Prompt types to track
PROMPT_TYPES = ['basic', 'formal', 'informal', 'detailed', 'short']

# Base directory
base_dir = 'eval_outputs'

# Dictionary to store raw results
# Structure: raw_results[model][task][metric][prompt_type] = list of {language, value}
raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

# Track missing data
missing_files = defaultdict(lambda: defaultdict(list))

print_task_order = ['ASR', 'SQA', 'ACHAP', 'TTS', 'MT', 'ST', 'TSUM', 'SSUM', 'S2ST']
print_model_order = ['phi_multimodal', 'qwen_omni']

# Process each model and task
print("Processing files...")
files_processed = 0
files_missing = 0

for task in print_task_order:
    for model in print_model_order:
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]

        task_langs = MODEL_TASK_LANGUAGES.get(model, {}).get(task, TASK_LANGUAGES.get(task, []))

        for lang in task_langs:
            file_path = Path(base_dir) / model / task / f'{lang}_eval.json'

            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    files_processed += 1

                    if 'results' in data and 'per_prompt_type' in data['results']:
                        prompt_data = data['results']['per_prompt_type']

                        for metric in task_metrics:
                            for prompt_type in PROMPT_TYPES:
                                if prompt_type not in prompt_data:
                                    continue

                                # Use all_prompt_modalities as the combined value across modalities
                                pt_entry = prompt_data[prompt_type]
                                if 'all_prompt_modalities' in pt_entry and metric in pt_entry['all_prompt_modalities']:
                                    value = pt_entry['all_prompt_modalities'][metric]['mean']
                                    raw_results[model][task][metric][prompt_type].append({
                                        'language': lang,
                                        'value': value
                                    })

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

# Calculate averages per prompt type
output_results = {}

for model in models:
    output_results[model] = {}

    for task in tasks:
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]

        output_results[model][task] = {'metrics': task_metrics}

        for metric in task_metrics:
            output_results[model][task][metric] = {}

            for prompt_type in PROMPT_TYPES:
                entries = raw_results[model][task][metric][prompt_type]
                if entries:
                    values = [item['value'] for item in entries]
                    output_results[model][task][metric][prompt_type] = {
                        'average': sum(values) / len(values),
                        'num_languages': len(values),
                        'languages': sorted([item['language'] for item in entries]),
                        'per_language': {item['language']: item['value'] for item in entries}
                    }
                else:
                    output_results[model][task][metric][prompt_type] = None

# Save to JSON
metadata = {
    'files_processed': files_processed,
    'files_missing': files_missing,
    'total_expected': total_expected,
    'task_metrics': TASK_METRICS,
    'task_languages': TASK_LANGUAGES,
    'prompt_types': PROMPT_TYPES,
    'note': 'Values are averaged over all modalities (all_prompt_modalities) per prompt type',
    'missing_files_by_model_task': {
        model: {
            task: langs for task, langs in tasks_dict.items() if langs
        } for model, tasks_dict in missing_files.items()
    }
}

full_output = {
    'metadata': metadata,
    'results': output_results
}

output_file = 'prompt_type_averages.json'
with open(output_file, 'w') as f:
    json.dump(full_output, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Print summary table
col_w = 12
header_cols = ['basic', 'formal', 'informal', 'detailed', 'short', 'langs', 'model']
total_width = 10 + 18 + col_w * len(PROMPT_TYPES) + 10 + 16

print("\n" + "=" * total_width)
print("SUMMARY: Average Scores by Prompt Type (across all modalities, across available languages)")
print("=" * total_width)
print(f"{'Task':<10} {'Metric':<18} " +
      " ".join(f"{pt:>{col_w}}" for pt in PROMPT_TYPES) +
      f" {'Langs':>8} {'Model':<15}")
print("-" * total_width)

for task in print_task_order:
    for model in print_model_order:
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]

        expected_langs = len(MODEL_TASK_LANGUAGES.get(model, {}).get(task, TASK_LANGUAGES.get(task, [])))
        if expected_langs == 0:
            continue

        for metric in task_metrics:
            row_values = []
            num_langs = 0

            for prompt_type in PROMPT_TYPES:
                entry = output_results[model][task][metric].get(prompt_type)
                if entry:
                    row_values.append(f"{entry['average']:.2f}")
                    if num_langs == 0:
                        num_langs = entry['num_languages']
                else:
                    row_values.append("N/A")

            lang_str = f"{num_langs}/{expected_langs}"
            print(f"{task:<10} {metric:<18} " +
                  " ".join(f"{v:>{col_w}}" for v in row_values) +
                  f" {lang_str:>8} {model:<15}")

print("=" * total_width)
print("\nNote: Values are averaged over all modalities (text, male audio, female audio) combined.")
print("      For WER: lower is better. For CometQE, BERTScore, and UTMOS: higher is better.")
print("      'Langs' shows how many languages have data / expected languages for that task.")