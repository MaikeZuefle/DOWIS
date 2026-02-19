import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from utils import TASK_LANGUAGES, MODEL_TASK_LANGUAGES, TASK_METRICS, MODEL_DISPLAY_NAMES, LOWER_IS_BETTER, METRIC_DISPLAY_NAMES

MONOLINGUAL_TASKS = ['ASR', 'SQA', 'TTS', 'ACHAP']
CROSSLINGUAL_TASKS = ['MT', 'ST', 'S2ST']

# SSUM and TSUM appear in BOTH plots, filtered by language
SPLIT_TASKS = ['SSUM', 'TSUM']
MONOLINGUAL_LANGS = {'SSUM': ['en'], 'TSUM': ['en']}
CROSSLINGUAL_LANGS = {
    'SSUM': [l for l in TASK_LANGUAGES.get('SSUM', []) if l != 'en'],
    'TSUM': [l for l in TASK_LANGUAGES.get('TSUM', []) if l != 'en'],
}

# Define the structure
models = ['qwen_omni', 'phi_multimodal']

# Prompt types (x-axis)
PROMPT_TYPES = ['basic', 'formal', 'informal', 'detailed', 'short']

# Base directory
base_dir = 'eval_outputs'

HEATMAP_METRIC_OVERRIDE = {
    'TTS': 'ASR-WER',
    'S2ST': 'ASR-COMET'
}

# Structure: results[model][task][prompt_type][language] = diff (sign-adjusted)
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

print("Processing files...")

for model in models:
    for task in TASK_LANGUAGES.keys():
        metric = HEATMAP_METRIC_OVERRIDE.get(task, TASK_METRICS.get(task, ''))
        task_langs = TASK_LANGUAGES.get(task, [])

        for lang in task_langs:
            file_path = Path(base_dir) / model / task / f'{lang}_eval.json'

            if not file_path.exists():
                print(f"⊗ {model}/{task}/{lang}: File not found")
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if 'results' not in data or 'per_prompt_type' not in data['results']:
                    print(f"⚠ {model}/{task}/{lang}: No per_prompt_type data")
                    continue

                prompt_data = data['results']['per_prompt_type']

                for prompt_type in PROMPT_TYPES:
                    if prompt_type not in prompt_data:
                        continue

                    pt_entry = prompt_data[prompt_type]

                    # Get text prompt value
                    text_val = None
                    if 'text_prompt' in pt_entry and metric in pt_entry['text_prompt']:
                        text_val = pt_entry['text_prompt'][metric]['mean']

                    # Get combined speech value (average of male + female)
                    f_val = None
                    m_val = None
                    if 'f_audio_prompt' in pt_entry and metric in pt_entry['f_audio_prompt']:
                        f_val = pt_entry['f_audio_prompt'][metric]['mean']
                    if 'm_audio_prompt' in pt_entry and metric in pt_entry['m_audio_prompt']:
                        m_val = pt_entry['m_audio_prompt'][metric]['mean']

                    if f_val is not None and m_val is not None:
                        speech_val = (f_val + m_val) / 2
                    elif f_val is not None:
                        speech_val = f_val
                    elif m_val is not None:
                        speech_val = m_val
                    else:
                        speech_val = None

                    if text_val is not None and speech_val is not None:
                        if metric in LOWER_IS_BETTER:
                            diff = speech_val - text_val
                        else:
                            diff = text_val - speech_val

                        results[model][task][prompt_type][lang] = diff
                        print(f"✓ {model}/{task}/{lang}/{prompt_type}: diff={diff:.2f}")

            except Exception as e:
                print(f"✗ {model}/{task}/{lang}: Error - {str(e)}")

print("\nGenerating heatmaps...")


def compute_row(model, task, prompt_type, lang_filter=None):
    """Return list of diff values for a task, optionally filtered to specific languages."""
    lang_diffs = results[model][task][prompt_type]
    if lang_filter is not None:
        lang_diffs = {k: v for k, v in lang_diffs.items() if k in lang_filter}
    return list(lang_diffs.values())


def create_prompt_heatmap(model, task_list, task_type_name, fig_height=8,
                          split_task_langs=None):
    """
    Heatmap with tasks on y-axis and prompt types on x-axis.
    Each cell = speech vs text diff averaged across (filtered) languages.

    split_task_langs: dict {task: [langs]} to restrict SSUM/TSUM to specific languages.
                      If None, uses all available languages for each task.
    """

    matrix = []
    task_labels = []
    annotation_matrix = []

    for task in task_list:
        if task not in TASK_LANGUAGES:
            continue

        metric = HEATMAP_METRIC_OVERRIDE.get(task, TASK_METRICS.get(task, ''))
        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)

        # Determine language filter for this task
        lang_filter = None
        if split_task_langs and task in split_task_langs:
            lang_filter = split_task_langs[task]
            if not lang_filter:
                # No languages available for this task in this plot, skip
                continue

        row = []
        annot_row = []

        for prompt_type in PROMPT_TYPES:
 
            lang_diffs = compute_row(model, task, prompt_type, lang_filter)
            
            if lang_diffs:
                avg_diff = np.mean(lang_diffs)
                row.append(avg_diff)
                annot_row.append(f"{avg_diff:.2f}")
            else:
                row.append(np.nan)
                annot_row.append("N/A")

        matrix.append(row)
        annotation_matrix.append(annot_row)
        task_labels.append((task, f"({metric_display})"))

    if not matrix:
        print(f"No data for {model} - {task_type_name}")
        return

    matrix = np.array(matrix, dtype=float)

    valid_data = matrix[~np.isnan(matrix)]
    if len(valid_data) == 0:
        print(f"No valid data for {model} - {task_type_name}")
        return

    actual_min = np.min(valid_data)
    actual_max = np.max(valid_data)
    print(f"  Data range for {model} - {task_type_name}: min={actual_min:.2f}, max={actual_max:.2f}")

    from matplotlib.colors import LinearSegmentedColormap

    if actual_min >= 0:
        colors = ['#FFFFFF', '#800080']
        vmin, vmax = 0, actual_max
        center = None
    elif actual_max <= 0:
        colors = ['#FFD700', '#FFFFFF']
        vmin, vmax = actual_min, 0
        center = None
    else:
        colors = ['#FFD700', '#FFFFFF', '#800080']
        vmin, vmax = actual_min, actual_max
        center = 0

    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

    n_tasks = len(task_labels)
    n_prompts = len(PROMPT_TYPES)
    fig, ax = plt.subplots(figsize=(max(8, n_prompts * 1.8), max(fig_height, n_tasks * 0.9)))

    heatmap_kwargs = {
        'xticklabels': PROMPT_TYPES,
        'yticklabels': [t for t, m in task_labels],
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
        'annot': np.array(annotation_matrix),
        'fmt': '',
        'annot_kws': {'fontsize': 14},
        'cbar_kws': {'label': 'Text − Speech'},
        'ax': ax,
        'linewidths': 0.5,
        'linecolor': 'gray',
        'mask': np.isnan(matrix),
    }

    if center is not None:
        heatmap_kwargs['center'] = center

    sns.heatmap(matrix, **heatmap_kwargs)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Add metric name below each task name tick label
    for i, (task_name, metric_name) in enumerate(task_labels):
        ax.text(-0.01, i + 0.75, metric_name,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=11, color='black',
                clip_on=False)

    cbar = ax.collections[0].colorbar
    cbar.set_label('Text − Speech', fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel('Prompt Type', fontsize=20, fontweight='bold')
    ax.set_ylabel('Task', fontsize=20, fontweight='bold', labelpad=35)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    model_display = MODEL_DISPLAY_NAMES.get(model, model)
    model_name = model_display.replace('.', '_').replace('-', '_').replace(' ', '_')
    filename = output_dir / f'heatmap_prompt_types_{model_name}_{task_type_name.lower().replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename}")
    plt.close()


# Generate heatmaps for each model
for model in models:
    print(f"\nGenerating heatmaps for {model}...")

    # Monolingual: standard mono tasks + SSUM/TSUM filtered to en-en only
    create_prompt_heatmap(
        model,
        MONOLINGUAL_TASKS + SPLIT_TASKS,
        "Monolingual Tasks",
        fig_height=5,
        split_task_langs=MONOLINGUAL_LANGS
    )

    # Crosslingual: standard cross tasks + SSUM/TSUM filtered to en-de, en-it
    create_prompt_heatmap(
        model,
        CROSSLINGUAL_TASKS + SPLIT_TASKS,
        "Crosslingual Tasks",
        fig_height=5,
        split_task_langs=CROSSLINGUAL_LANGS
    )

print("\n✓ All heatmaps generated!")
print("\nColor interpretation:")
print("  🟣 Purple (positive values): Text prompt performs BETTER")
print("  🟡 Yellow (negative values): Speech prompts perform BETTER")
print("  ⚪ White (near zero): Similar performance")
print("  Values averaged across languages for each task × prompt type cell")
print("\nSSUM/TSUM language split:")
print(f"  Monolingual plot: {MONOLINGUAL_LANGS}")
print(f"  Crosslingual plot: {CROSSLINGUAL_LANGS}")