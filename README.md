# Do What I Say: A Spoken Prompt Dataset for Instruction-Following
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-DOWIS-yellow)](https://huggingface.co/datasets/maikezu/dowis)


This repository contains code for the DOWIS prompt project. 
The dataset can be found on HuggingFace at [maikezu/dowis](https://huggingface.co/datasets/maikezu/dowis).

> **TL;DR** — DOWIS is a multilingual dataset of human-recorded spoken and written instruction prompts, designed to enable realistic evaluation of Speech Large Language Models across 11 tasks and 12 languages.


> 🆕 **New:** DOWIS now also contains spoken and written prompts in Albanian (sq), and for the tasks LIPREAD and SLU!
> 
---
### Paper Abstract

Speech Large Language Models (SLLMs) have rapidly expanded, supporting a wide range of tasks. These models are typically evaluated using text prompts, which may not reflect real-world scenarios where users interact with speech. 
To address this gap, we introduce DoWhatISay (DOWIS), a multilingual dataset of human-recorded spoken and written prompts designed to pair with any existing benchmark for realistic evaluation of SLLMs under spoken instruction conditions. 
Spanning 9 tasks and 11 languages, it provides 10 prompt variants per task-language pair, across five styles.
Using DOWIS, we benchmark state-of-the-art SLLMs, analyzing the interplay between prompt modality, style, language, and task type. Results show that text prompts consistently outperform spoken prompts, particularly for low-resource and cross-lingual settings. Only for tasks with speech output, spoken prompts do  close the gap, highlighting the need for speech-based prompting in SLLM evaluation.

> **Note:** DOWIS has since been extended to 11 tasks and 12 languages, including Albanian (sq) and the tasks LIPREAD and SLU.
---

The code includes inference and evaluation code for 

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

## Inference

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

## Evaluation
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

## Citation

If you use this work, please cite:
```bibtex
@misc{züfle2026isayspokenprompt,
      title={Do What I Say: A Spoken Prompt Dataset for Instruction-Following}, 
      author={Maike Züfle and
              Sara Papi and
              Fabian Retkowski and
              Szymon Mazurek and
              Marek Kasztelnik and
              Alexander Waibel and
              Luisa Bentivogli and
              Jan Niehues},
      year={2026},
      eprint={2603.09881},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.09881}}
```

