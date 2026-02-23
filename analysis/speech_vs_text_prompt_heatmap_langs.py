import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from utils import TASK_LANGUAGES, MODEL_TASK_LANGUAGES, TASK_METRICS, MODEL_DISPLAY_NAMES, LOWER_IS_BETTER, METRIC_DISPLAY_NAMES

# Task definitions
MONOLINGUAL_TASKS = ['ASR']
MULTILINGUAL_TASKS = ['MT', 'ST']

# Define the structure
models = ['qwen_omni'] #, 'phi_multimodal']

# Base directory
base_dir = 'eval_outputs'

# Dictionary to store results
# Structure: results[model][task][language] = {'text': value, 'combined': value, 'diff': value}
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

HEATMAP_METRIC_OVERRIDE = {
    'TTS': 'ASR-WER',
    'S2ST': 'ASR-COMET',
    'ACHAP': 'CollarF1'
}

print("Processing files...")

for model in models:
    for task in TASK_LANGUAGES.keys():
        metric = HEATMAP_METRIC_OVERRIDE.get(task, TASK_METRICS.get(task, ''))
        task_langs = TASK_LANGUAGES.get(task, [])

        for lang in task_langs:
            file_path = Path(base_dir) / model / task / f'{lang}_eval.json'

            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    if 'results' in data and 'per_modality' in data['results']:
                        modalities_data = data['results']['per_modality']

                        text_val = None
                        combined_val = None

                        # Get text prompt value
                        if 'text_prompt' in modalities_data and metric in modalities_data['text_prompt']:
                            text_val = modalities_data['text_prompt'][metric]['mean']

                        # Calculate combined from male and female
                        f_val = None
                        m_val = None
                        if 'f_audio_prompt' in modalities_data and metric in modalities_data['f_audio_prompt']:
                            f_val = modalities_data['f_audio_prompt'][metric]['mean']
                        if 'm_audio_prompt' in modalities_data and metric in modalities_data['m_audio_prompt']:
                            m_val = modalities_data['m_audio_prompt'][metric]['mean']

                        # Calculate combined as average of male and female
                        if f_val is not None and m_val is not None:
                            combined_val = (f_val + m_val) / 2
                        elif f_val is not None:
                            combined_val = f_val
                        elif m_val is not None:
                            combined_val = m_val

                        # Calculate difference
                        if text_val is not None and combined_val is not None:
                            # For metrics where lower is better, flip the sign
                            # So positive diff means text is better
                            if metric in LOWER_IS_BETTER:
                                diff = combined_val - text_val  # If text has lower WER, diff is positive
                            else:
                                diff = text_val - combined_val  # If text has higher score, diff is positive

                            results[model][task][lang] = {
                                'text': text_val,
                                'combined': combined_val,
                                'diff': diff
                            }
                            print(f"✓ {model}/{task}/{lang}: text={text_val:.2f}, combined={combined_val:.2f}, diff={diff:.2f}")

                except Exception as e:
                    print(f"✗ {model}/{task}/{lang}: Error - {str(e)}")

print("\nGenerating heatmaps...")


def create_heatmap(model, task_list, task_type_name, fig_height=8):
    """Create a heatmap for a specific model and set of tasks"""

    # Get all languages across these tasks
    all_langs = sorted(list(set(lang for task in task_list for lang in TASK_LANGUAGES.get(task, []))))

    # Create matrix for differences
    matrix = []
    task_labels = []
    mask_unsupported = []  # Track which cells are unsupported languages

    for task in task_list:
        if task not in TASK_LANGUAGES:
            continue

        row = []
        mask_row = []
        task_langs = TASK_LANGUAGES[task]

        for lang in all_langs:
            if lang not in task_langs:
                row.append(0)          # Placeholder value
                mask_row.append(True)  # Mark as unsupported (will be grey)
            elif lang in results[model][task]:
                row.append(results[model][task][lang]['diff'])
                mask_row.append(False)  # Has data
            else:
                row.append(np.nan)     # Missing data (white)
                mask_row.append(False)

        matrix.append(row)
        mask_unsupported.append(mask_row)

        # Build task label as (task_name, metric_display) tuple
        metric = HEATMAP_METRIC_OVERRIDE.get(task, TASK_METRICS.get(task, ''))
        metric_display = METRIC_DISPLAY_NAMES.get(metric, metric)
        task_labels.append((task, f"({metric_display})"))

    if not matrix:
        print(f"No data for {model} - {task_type_name}")
        return

    matrix = np.array(matrix)
    mask_unsupported = np.array(mask_unsupported)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_langs) * 0.8), fig_height))

    # Calculate vmin and vmax based on actual data
    valid_data = matrix[~mask_unsupported & ~np.isnan(matrix)]
    if len(valid_data) > 0:
        actual_min = np.min(valid_data)
        actual_max = np.max(valid_data)

        print(f"  Data range for {model} - {task_type_name}: min={actual_min:.2f}, max={actual_max:.2f}")

        from matplotlib.colors import LinearSegmentedColormap

        if actual_min >= 0:
            colors = ['#FFFFFF', '#800080']
            vmin, vmax = 0, actual_max
            center = None
            print(f"  Using WHITE->PURPLE colormap (all values positive)")
        elif actual_max <= 0:
            colors = ['#FFD700', '#FFFFFF']
            vmin, vmax = actual_min, 0
            center = None
            print(f"  Using YELLOW->WHITE colormap (all values negative)")
        else:
            colors = ['#FFD700', '#FFFFFF', '#800080']
            vmin, vmax = actual_min, actual_max
            center = 0
            print(f"  Using YELLOW->WHITE->PURPLE colormap (mixed values, actual range)")

        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    else:
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#FFD700', '#FFFFFF', '#800080']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        vmin, vmax = -1, 1
        center = 0
        print(f"  No valid data - using default colormap")

    # Create heatmap — yticklabels is just task names, metric added separately below
    heatmap_kwargs = {
        'xticklabels': all_langs,
        'yticklabels': [t for t, m in task_labels],
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
        'annot': True,
        'fmt': '.2f',
        'annot_kws': {'fontsize': 14},
        'cbar_kws': {'label': 'Text - Speech Prompts'},
        'ax': ax,
        'linewidths': 0.5,
        'linecolor': 'gray',
        'mask': (np.isnan(matrix) | mask_unsupported)
    }

    if center is not None:
        heatmap_kwargs['center'] = center

    sns.heatmap(matrix, **heatmap_kwargs)

    # Set font sizes explicitly on axes (not via rcParams, which scales with figure size)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Add metric name in smaller black text below each task name tick label
    for i, (task_name, metric_name) in enumerate(task_labels):
        label = ax.get_yticklabels()[i]
        x, y = label.get_position()
        ax.text(x, y + 0.15, metric_name,
                transform=label.get_transform(),
                ha='right', va='center', fontsize=11, color='black')

    cbar = ax.collections[0].colorbar
    cbar.set_label('Text - Speech Prompts', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Add grey overlay for unsupported languages
    for i in range(len(task_list)):
        for j in range(len(all_langs)):
            if mask_unsupported[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           facecolor='lightgrey', edgecolor='gray', linewidth=0.5))
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=14, color='darkgrey')

    ax.set_xlabel('Language', fontsize=20, fontweight='bold')
    ax.set_ylabel('Task', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    model_name = MODEL_DISPLAY_NAMES.get(model, model).replace('.', '_').replace('-', '_')
    filename = output_dir / f'heatmaps/heatmap_{model_name}_{task_type_name.lower().replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename}")
    plt.close()


# Generate heatmaps for each model
for model in models:
    print(f"\nGenerating heatmaps for {model}...")

    # Monolingual tasks
    create_heatmap(model, MONOLINGUAL_TASKS, "Monolingual Tasks", fig_height=5)

    # Multilingual tasks
    create_heatmap(model, MULTILINGUAL_TASKS, "Multilingual Tasks", fig_height=5)

print("\n✓ All heatmaps generated!")
print("\nColor interpretation:")
print("  🟣 Purple (positive values): Text prompt performs BETTER")
print("  🟡 Yellow (negative values): Speech prompts perform BETTER")
print("  ⚪ White (near zero): Similar performance")
print("  ⬜ Grey (N/A): Language not supported by task")