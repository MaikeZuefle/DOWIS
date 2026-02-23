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

output_file = 'analysis/result_json/prompt_type_averages.json'
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

# Metrics where lower is better (will be inverted for colour scaling)
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


def fmt_value(entry, metric):
    if entry is None:
        return ''
    scale = METRIC_SCALE.get(metric, 1)
    return f"{entry['average'] * scale:.2f}"


def collect_metric_values(output_results, task, metric, model, prompt_types):
    """Collect all numeric values for a given task+metric+model across all prompt types."""
    values = []
    for prompt_type in prompt_types:
        entry = output_results[model][task][metric].get(prompt_type)
        if entry is not None:
            scale = METRIC_SCALE.get(metric, 1)
            values.append(entry['average'] * scale)
    return values


def compute_alpha(value, min_val, max_val, lower_is_better):
    """Map a value to an alpha intensity in [ALPHA_MIN, ALPHA_MAX].
    Better values get higher alpha (darker colour)."""
    if max_val == min_val:
        return (ALPHA_MIN + ALPHA_MAX) / 2
    norm = (value - min_val) / (max_val - min_val)
    if lower_is_better:
        norm = 1.0 - norm
    return ALPHA_MIN + norm * (ALPHA_MAX - ALPHA_MIN)


def coloured_cell(value_str, alpha, colour_name):
    """Wrap a cell value with a \cellcolor command."""
    if value_str == '':
        return ''
    intensity = int(round(max(0.0, min(1.0, alpha)) * 100))  # clamp to [0, 100]
    return rf'\cellcolor{{{colour_name}!{intensity}}} {value_str}'


def build_latex_table(output_results, print_task_order, print_model_order,
                      TASK_METRICS, MODEL_TASK_LANGUAGES, TASK_LANGUAGES):

    lines = []

    lines.append(r'% Add to your preamble:')
    lines.append(r'% \usepackage[table]{xcolor}')
    lines.append(r'% \definecolor{taskgreen}{RGB}{180, 220, 180}')
    lines.append(r'% \definecolor{taskblue}{RGB}{180, 210, 230}')
    lines.append(r'')
    lines.append(r'\begin{table*}[]')
    lines.append(r'    \centering')
    lines.append(r'    \footnotesize')
    lines.append(r'    \begin{tabular}{llcccccc}')
    lines.append(r'     \toprule')
    lines.append(r'     \multirow{2}{*}{\textbf{Task}} & \multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{5}{c}{\textbf{Prompt Type}} \\')
    lines.append(r'     \cmidrule(lr){4-8}')
    lines.append(r'     &&& \textbf{Basic} & \textbf{Formal} & \textbf{Inform.} & \textbf{Detail.} & \textbf{Short} \\')
    lines.append(r'     \midrule')

    num_tasks = len(print_task_order)

    for task_idx, task in enumerate(print_task_order):
        task_metrics = TASK_METRICS.get(task, 'wer')
        if isinstance(task_metrics, str):
            task_metrics = [task_metrics]

        task_models = ['qwen_omni'] if task in QWEN_ONLY_TASKS else print_model_order
        num_rows = len(task_metrics) * len(task_models)

        colour_name = TASK_COLOURS[task_idx % 2]


        # Pre-compute per-metric value ranges, anchored to the better model
        metric_ranges = {}
        for metric in task_metrics:
            lower_better = metric in LOWER_IS_BETTER
            best_min, best_max = None, None

            for model in task_models:
                values = collect_metric_values(output_results, task, metric, model, PROMPT_TYPES)
                if not values:
                    continue
                model_min, model_max = min(values), max(values)
                # The "better" model is the one with the better average
                model_avg = sum(values) / len(values)
                if best_max is None:
                    best_min, best_max = model_min, model_max
                    best_avg = model_avg
                else:
                    is_better = (model_avg < best_avg) if lower_better else (model_avg > best_avg)
                    if is_better:
                        best_min, best_max = model_min, model_max
                        best_avg = model_avg

            # Use the better model's range for both models
            shared_range = (best_min, best_max) if best_min is not None else (0, 1)
            metric_ranges[metric] = {model: shared_range for model in task_models}

        for metric_idx, metric in enumerate(task_metrics):
            metric_latex = METRIC_LATEX.get(metric, metric)
            lower_better = metric in LOWER_IS_BETTER

            for model_idx, model in enumerate(task_models):
                # ← fix: look up ranges inside the model loop
                min_val, max_val = metric_ranges[metric][model]

                model_name = MODEL_LATEX.get(model, model)
                if model == 'phi_multimodal' and task in PHI_FOOTNOTE_TASKS:
                    model_name += '*'

                # Build value cells with colour
                cells = []
                for prompt_type in PROMPT_TYPES:
                    entry = output_results[model][task][metric].get(prompt_type)
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
                lines.append(r'     \cmidrule(lr){1-8}')

    lines.append(r'     \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r"    \caption{The impact of prompt type across different tasks. Results are averaged over all modalities and available languages. *Phi only supports the languages 'en', 'de', 'fr', 'it', 'es', 'pt' for speech in input, so we only report results for these languages for ASR.}")
    lines.append(r'    \label{tab:prompt_type}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


latex_table = build_latex_table(
    output_results, print_task_order, print_model_order,
    TASK_METRICS, MODEL_TASK_LANGUAGES, TASK_LANGUAGES
)

latex_output_file = 'analysis/latex/prompt_type_table.tex'
with open(latex_output_file, 'w') as f:
    f.write(latex_table)

print(f"\n✓ LaTeX table saved to: {latex_output_file}")