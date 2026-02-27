import json
from pathlib import Path
from collections import defaultdict
from utils import TASK_LANGUAGES, MODEL_TASK_LANGUAGES, TASK_METRICS

# Define the structure
models = ['qwen_omni', 'phi_multimodal']
tasks = ['ASR', 'MT', 'S2ST', 'SSUM', 'TSUM', 'ST', 'TTS', 'ACHAP', 'SQA']

# languages where we have both speakers
GENDERED_LANGUAGES = {'de', 'en', 'it', 'cs', 'es', 'fr'}

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
            
            # Female audio prompt average (only gendered languages)
            if 'f_audio_prompt' in raw_results[model][task][metric] and len(raw_results[model][task][metric]['f_audio_prompt']) > 0:
                values = [item['value'] for item in raw_results[model][task][metric]['f_audio_prompt']
                          if item['language'] in GENDERED_LANGUAGES]
                if values:
                    output_results[model][task][metric]['f_audio_prompt'] = {
                        'average': sum(values) / len(values),
                        'num_languages': len(values),
                        'languages': sorted([item['language'] for item in raw_results[model][task][metric]['f_audio_prompt']
                                             if item['language'] in GENDERED_LANGUAGES]),
                        'per_language': {item['language']: item['value'] for item in raw_results[model][task][metric]['f_audio_prompt']
                                         if item['language'] in GENDERED_LANGUAGES}
                    }
                else:
                    output_results[model][task][metric]['f_audio_prompt'] = None
            else:
                output_results[model][task][metric]['f_audio_prompt'] = None
            
            # Male audio prompt average (only gendered languages)
            if 'm_audio_prompt' in raw_results[model][task][metric] and len(raw_results[model][task][metric]['m_audio_prompt']) > 0:
                values = [item['value'] for item in raw_results[model][task][metric]['m_audio_prompt']
                          if item['language'] in GENDERED_LANGUAGES]
                if values:
                    output_results[model][task][metric]['m_audio_prompt'] = {
                        'average': sum(values) / len(values),
                        'num_languages': len(values),
                        'languages': sorted([item['language'] for item in raw_results[model][task][metric]['m_audio_prompt']
                                             if item['language'] in GENDERED_LANGUAGES]),
                        'per_language': {item['language']: item['value'] for item in raw_results[model][task][metric]['m_audio_prompt']
                                         if item['language'] in GENDERED_LANGUAGES}
                    }
                else:
                    output_results[model][task][metric]['m_audio_prompt'] = None
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
output_file = 'analysis/result_json/modality_averages.json'
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

# =============================================================================
# LaTeX Table Generation
# =============================================================================

METRIC_LATEX = {
    'wer':           r'WER $\downarrow$',
    'CometQE':       r'Comet $\uparrow$',
    'ASR-COMET':     r'COM.$_{\text{ASR}}$ $\uparrow$',
    'BERTScore_F1':  r'BERTS. $\uparrow$',
    'UTMOS':         r'UTMOS $\uparrow$',
    'ASR-WER':       r'WER$_{\text{ASR}}$ $\downarrow$',
    'CollarF1':      r'CollarF1 $\uparrow$',
    'GC-BS':         r'GC-BS $\uparrow$',
}

LOWER_IS_BETTER = {'wer', 'ASR-WER'}

MODEL_LATEX = {
    'phi_multimodal': 'Phi',
    'qwen_omni':      'Qwen',
}

QWEN_ONLY_TASKS = {'TTS', 'S2ST'}
PHI_FOOTNOTE_TASKS = {'ASR'}
MIDRULE_AFTER = {'TTS'}

METRIC_SCALE = {
    'CollarF1': 100,
    'GC-BS':    100,
}

TASK_COLOURS = ['taskgreen', 'taskblue']

ALPHA_MIN = 0.20
ALPHA_MAX = 0.80

# Modalities in display order: text, combined audio, male, female
MODALITIES = ['text_prompt', 'audio_prompt_combined', 'm_audio_prompt', 'f_audio_prompt']


def fmt_value(entry, metric):
    if entry is None:
        return ''
    scale = METRIC_SCALE.get(metric, 1)
    return f"{entry['average'] * scale:.2f}"


def collect_metric_values_modality(output_results, task, metric, model, modalities):
    """Collect all numeric values for a given task+metric+model across all modalities."""
    values = []
    for modality in modalities:
        entry = output_results[model][task][metric].get(modality)
        if entry is not None:
            scale = METRIC_SCALE.get(metric, 1)
            values.append(entry['average'] * scale)
    return values


def compute_alpha(value, min_val, max_val, lower_is_better):
    if max_val == min_val:
        return (ALPHA_MIN + ALPHA_MAX) / 2
    norm = (value - min_val) / (max_val - min_val)
    if lower_is_better:
        norm = 1.0 - norm
    return ALPHA_MIN + norm * (ALPHA_MAX - ALPHA_MIN)


def coloured_cell(value_str, alpha, colour_name):
    if value_str == '':
        return ''
    intensity = int(round(max(0.0, min(1.0, alpha)) * 100))
    return rf'\cellcolor{{{colour_name}!{intensity}}} {value_str}'


def build_latex_table_modality(output_results, print_task_order, print_model_order,
                               TASK_METRICS, MODEL_TASK_LANGUAGES, TASK_LANGUAGES):

    lines = []

    lines.append(r'% Add to your preamble:')
    lines.append(r'% \usepackage[table]{xcolor}')
    lines.append(r'% \definecolor{taskgreen}{RGB}{180, 220, 180}')
    lines.append(r'% \definecolor{taskblue}{RGB}{180, 210, 230}')
    lines.append(r'')
    lines.append(r'\begin{table}[]')
    lines.append(r'    \centering')
    lines.append(r'    \footnotesize')
    lines.append(r'    \begin{tabular}{p{0.6cm}p{1.3cm}p{0.6cm}p{0.7cm}p{0.7cm}p{0.7cm}p{0.7cm}}')
    lines.append(r'     \toprule')
    lines.append(r'     \multirow{2}{*}{\textbf{Task}} & \multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Model}} & \textbf{Text} & \multicolumn{3}{c}{\textbf{Speech Prompt}} \\')
    lines.append(r'     &&& \textbf{Prompt} & \multicolumn{1}{c}{\textbf{All}} & \multicolumn{1}{c}{\textbf{Male}} & \multicolumn{1}{c}{\textbf{Fem.}} \\')
    lines.append(r'     \midrule')

    num_tasks = len(print_task_order)

    for task_idx, task in enumerate(print_task_order):
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]

        task_models = ['qwen_omni'] if task in QWEN_ONLY_TASKS else print_model_order
        num_rows = len(task_metrics) * len(task_models)

        colour_name = TASK_COLOURS[task_idx % 2]

        # Pre-compute per-metric ranges anchored to the better model
        metric_ranges = {}
        for metric in task_metrics:
            lower_better = metric in LOWER_IS_BETTER
            best_min, best_max, best_avg = None, None, None

            for model in task_models:
                values = collect_metric_values_modality(output_results, task, metric, model, MODALITIES)
                if not values:
                    continue
                model_avg = sum(values) / len(values)
                if best_avg is None:
                    best_min, best_max, best_avg = min(values), max(values), model_avg
                else:
                    is_better = (model_avg < best_avg) if lower_better else (model_avg > best_avg)
                    if is_better:
                        best_min, best_max, best_avg = min(values), max(values), model_avg

            shared_range = (best_min, best_max) if best_min is not None else (0, 1)
            metric_ranges[metric] = {model: shared_range for model in task_models}

        for metric_idx, metric in enumerate(task_metrics):
            metric_latex = METRIC_LATEX.get(metric, metric)
            lower_better = metric in LOWER_IS_BETTER

            for model_idx, model in enumerate(task_models):
                min_val, max_val = metric_ranges[metric][model]

                model_name = MODEL_LATEX.get(model, model)
                if model == 'phi_multimodal' and task in PHI_FOOTNOTE_TASKS:
                    model_name += '*'

                # Build 4 value cells: text, combined, male, female
                cells = []
                for modality in MODALITIES:
                    entry = output_results[model][task][metric].get(modality)
                    value_str = fmt_value(entry, metric)
                    if value_str != '':
                        scale = METRIC_SCALE.get(metric, 1)
                        raw_value = entry['average'] * scale
                        alpha = compute_alpha(raw_value, min_val, max_val, lower_better)
                        cells.append(coloured_cell(value_str, alpha, colour_name))
                    else:
                        cells.append('')
                cells_str = ' & '.join(cells)

                # Task label (first row of block only)
                global_row = metric_idx * len(task_models) + model_idx
                if global_row == 0:
                    task_col = rf'\multirow{{{num_rows}}}{{*}}{{\textbf{{{task}}}}}'
                else:
                    task_col = ''

                # Metric label (first model row per metric only)
                if model_idx == 0:
                    if len(task_models) > 1:
                        metric_col = rf'\multirow{{{len(task_models)}}}{{*}}{{{metric_latex}}}'
                    else:
                        metric_col = metric_latex
                else:
                    metric_col = ''

                row = f'     {task_col} & {metric_col} & {model_name} & {cells_str} \\\\'
                lines.append(row)

        # Separator after task block
        if task_idx < num_tasks - 1:
            if task in MIDRULE_AFTER:
                lines.append(r'     \midrule')
            else:
                lines.append(r'     \cmidrule(lr){1-7}')

    lines.append(r'     \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r"    \caption{The impact of speech vs. text prompts across different tasks. The results are averaged over different prompt types and languages. *Phi only supports the languages 'en', 'de', 'fr', 'it', 'es', 'pt' for speech in input, so we only report results for these languages for ASR.}")
    lines.append(r'    \label{tab:modality}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


latex_table = build_latex_table_modality(
    output_results, print_task_order, print_model_order,
    TASK_METRICS, MODEL_TASK_LANGUAGES, TASK_LANGUAGES
)

latex_output_file = 'analysis/latex/modality_table.tex'
with open(latex_output_file, 'w') as f:
    f.write(latex_table)

print(f"\n✓ LaTeX table saved to: {latex_output_file}")