# DOWIS (Do What I Say)

## 🌍 Supported Languages

The following languages are currently supported:

```
alb, cs, de, en, es, fr, hu, it, lv, nl, pt, ru, sv
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

## 🚀 Usage

The main entry point of the project is `main.py`, which can be executed from the command line.

### Command-line Arguments

| Argument | Description | Choices | Default |
|--------|-------------|---------|---------|
| `--lang` | Language to process | alb, cs, de, en, es, fr, hu, it, lv, nl, pt, ru, sv | alb |
| `--modality` | Prompt modality | text, audio | text |
| `--task` | Task to run | ACHAP, ASR, MT, S2ST, SLU, SQA, SSUM, ST, TSUM, TTS | ACHAP |
| `--model` | Model to use | phi_multimodal, qwen_omni | phi_multimodal |
| `--out_folder` | Output directory | any path | generated_output |

---

### Example Command

```bash
python main.py \
  --lang de \
  --modality audio \
  --model phi_multimodal \
  --task ASR \
  --out_folder outputs
```

---

## 📂 Output

Generated outputs are stored in the specified output folder as JSON files. Each output contains:

- Reference text
- Prompt type
- Model outputs per prompt style and modality

---

If you use or extend DOWIS in your research, please consider citing the project.

