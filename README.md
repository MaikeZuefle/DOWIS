# DOWIS (Do What I Say)

## 🌍 Supported Languages

The following languages are currently supported:

```
cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv
```

---

## 🎧 Modalities

DOWIS supports two prompt and task modalities:

- `text`  – written text prompts
- `audio` – spoken prompts (e.g., WAV files)

---

## 🧠 Tasks

The following tasks are implemented:

| Task Code | Description |
|---------|-------------|
| ACHAP | Audio Chaptering |
| ASR | Automatic Speech Recognition |
| MT | Machine Translation |
| S2ST | Speech-to-Speech Translation |
| SLU | Spoken Language Understanding |
| SQA | Spoken Question Answering |
| SSUM | Speech Summarization |
| ST | Speech Translation |
| TSUM | Text Summarization |
| TTS | Text-to-Speech |

---

## 🤖 Models

The following models are supported:

- `phi_multimodal`
- `qwen_omni`

---

## 🚀 Inference

The main entry point of the project is `main.py`, which can be executed from the command line.

### Command-line Arguments

| Argument | Description | Choices                                             | Default          |
|--------|-------------|-----------------------------------------------------|------------------|
| `--lang` | Language to process | cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv      | cs               |
| `--task` | Task to run | ACHAP, ASR, MT, S2ST, SLU, SQA, SSUM, ST, TSUM, TTS | ACHAP            |
| `--model` | Model to use | phi_multimodal, qwen_omni                           | phi_multimodal   |
| `--out_folder` | Output directory | any path                                            | generated_output |

---

### Example Command

```bash
python main.py \
  --lang de \
  --model phi_multimodal \
  --task ASR \
  --out_folder outputs
```

---

### 📂 Inference Output

Generated outputs are stored in the specified output folder as JSON files. Each output contains:

- Reference text
- Prompt type
- Model outputs per prompt style and modality

---

## 📊 Evaluation
The evaluation script `eval_outputs.py` computes metrics on the generated predictions from `main.py`.

### Command-line Arguments
| Argument | Description | Choices | Default |
|--------|-------------|---------|---------|
| `--lang` | Language to evaluate | cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv | cs |
| `--task` | Task to evaluate | ACHAP, ASR, MT, S2ST, SLU, SQA, SSUM, ST, TSUM, TTS | ACHAP |
| `--model` | Model to evaluate | phi_multimodal, qwen_omni | phi_multimodal |
| `--out_folder` | Output directory for evaluation results | any path | evaluation_results |
| `--predictions_folder` | Folder containing predictions (optional) | any path | same as `--out_folder` |

---

### Example Commands
```bash
python eval_outputs.py \
  --lang de \
  --model phi_multimodal \
  --task ASR \
  --predictions_folder outputs \
  --out_folder evaluation_results
```

---

### 📈 Evaluation Output
Evaluation results are stored as JSON files containing:
- **Summary statistics**: Unique samples, total evaluations, prompt types, and modalities
- **Per prompt type results**: Metrics for each prompt style (basic, formal, informal, detailed, short)
- **Per modality results**: Metrics aggregated by prompt modality (text, female audio, male audio)
- **Overall results**: Global metrics across all prompt types and modalities

#### Example Output Structure
```json
{
  "task": "ASR",
  "language": "en",
  "model": "qwen_omni",
  "summary_stats": {
    "unique_samples": 1294,
    "total_evaluations": 19410,
    "num_prompt_types": 5,
    "num_modalities": 3,
    "evaluations_per_prompt_type": {
      "basic": 3882,
      "formal": 3882,
      ...
    },
    "evaluations_per_modality": {
      "text_prompt": 6470,
      "f_audio_prompt": 6470,
      "m_audio_prompt": 6470
    }
  },
  "results": {
    "per_prompt_type": { ... },
    "per_modality": { ... },
    "overall": {
      "wer": {
        "mean": 12.355,
        "count": 19410
      }
    }
  }
}
```

---

### 📋 Task-Specific Metrics
Different tasks use different evaluation metrics:
- **ASR,**: WER 
- **MT, ST**: COMET QE
- **SQA, TSUM, SSUM**: BERTScore
- **TTS**: ASR-WER, UTMOS
- **S2ST**: ASR-COMET, UTMOS
- **ACHAP**: TODO


---


If you use or extend DOWIS in your research, please consider citing the project.

