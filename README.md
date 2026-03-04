# DOWIS (Do What I Say)

This repository contains code for the DOWIS prompt project. 
It includes inference and evaluation code for 

- 🤖 the models  `phi_multimodal` and `qwen_omni`
- 🌍 the languages  cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv
- 🎧 the prompt modalities `text`  and `audio`
- 🧠 the tasks below:


  | Task Code | Description |
  |---------|-------------|
  | ACHAP | Audio Chaptering |
  | ASR | Automatic Speech Recognition |
  | MT | Machine Translation |
  | S2ST | Speech-to-Speech Translation |
  | SQA | Spoken Question Answering |
  | SSUM | Speech Summarization |
  | ST | Speech Translation |
  | TSUM | Text Summarization |
  | TTS | Text-to-Speech |


We also release the model outputs in the [`outputs`](./outputs) folder.

---

## 🚀 Inference

Run [`main.py`](main.py) with the following arguments to start inference:

```bash
python main.py \
  --lang de \
  --model phi_multimodal \
  --task ASR \
  --out_folder outputs
```


| Argument | Description | Choices                                             | Default          |
|--------|-------------|-----------------------------------------------------|------------------|
| `--lang` | Language to process | cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv      | cs               |
| `--task` | Task to run | ACHAP, ASR, MT, S2ST, SQA, SSUM, ST, TSUM, TTS | ACHAP            |
| `--model` | Model to use | phi_multimodal, qwen_omni                           | phi_multimodal   |
| `--out_folder` | Output directory | any path                                            | generated_output |

---
📂 Generated outputs are stored in the specified output folder as JSON files. Each output contains:
- Reference text
- Prompt type
- Model outputs per prompt style and modality

---

## 📊 Evaluation
The evaluation script [`eval_outputs.py`](eval_outputs.py) computes metrics on the generated predictions.
```bash
python eval_outputs.py \
  --lang de \
  --model phi_multimodal \
  --task ASR \
  --predictions_folder outputs \
  --out_folder evaluation_results
```

| Argument | Description | Choices | Default |
|--------|-------------|---------|---------|
| `--lang` | Language to evaluate | cs, de, en, es, fr, hu, it, nl, pt, ru, sq, sv | cs |
| `--task` | Task to evaluate | ACHAP, ASR, MT, S2ST, SQA, SSUM, ST, TSUM, TTS | ACHAP |
| `--model` | Model to evaluate | phi_multimodal, qwen_omni | phi_multimodal |
| `--out_folder` | Output directory for evaluation results | any path | evaluation_results |
| `--predictions_folder` | Folder containing predictions (optional) | any path | same as `--out_folder` |

---
📈  Evaluation results are stored as JSON files containing:
- **Summary statistics**: Unique samples, total evaluations, prompt types, and modalities
- **Per prompt type results**: Metrics for each prompt style (basic, formal, informal, detailed, short)
- **Per modality results**: Metrics aggregated by prompt modality (text, female audio, male audio)
- **Overall results**: Global metrics across all prompt types and modalities


---

If you use or extend DOWIS in your research, please consider citing the project.
