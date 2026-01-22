import argparse
import logging
import os
from tqdm import tqdm
from transformers import set_seed
import json

# import multimodal models
from models.qwen_omni import generate as generate_qwen_omni
from models.qwen_omni import load_model as load_qwen_omni
from models.phi_multimodal import generate as generate_phi_multimodal
from models.phi_multimodal import load_model as load_phi_multimodal

# import data
from data.achap import load_achap
from data.asr import load_asr
from data.mt import load_mt
from data.s2st import load_s2st
from data.slu import load_slu
from data.sqa import load_sqa
from data.ssum import load_ssum
from data.st import load_st
from data.tsum import load_tsum
from data.tts import load_tts

# utils
from utils import set_up_logging

# setting seed for reproducibilty
set_seed(42)


def load_model(model_name):
    if model_name == "phi_multimodal":
        model = load_phi_multimodal()
        generate_func = generate_phi_multimodal
    elif model_name == "qwen_omni":
        model = load_qwen_omni()
        generate_func = generate_qwen_omni
    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")
    return model, generate_func

def load_data(task, language):
    task = task.lower()
    if task == "achap": data = load_achap(language)
    elif task == "asr": data = load_asr(language)
    elif task == "mt": data = load_mt(language)
    elif task == "s2st": data = load_s2st(language)
    elif task == "slu": data = load_slu(language)
    elif task == "sqa": data = load_sqa(language)
    elif task == "ssum": data = load_ssum(language)
    elif task == "st": data = load_st(language)
    elif task == "tsum": data = load_tsum(language)
    elif task == "tts": data = load_tts(language)
    else:
        raise NotImplementedError()
    return data

def load_prompt(task, language):
    prompts_path = f"prompts/prompts_{language}.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        if not task.lower() in prompts.keys():
            raise KeyError(f"Task {task} does not have prompts in {prompts_path}.")
    return prompts[task.lower()]


def main(out_folder, model, task, modality, lang):

    output_file_path = f"{out_folder}/{model}/{task}/{modality}_{lang}.json"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    set_up_logging(output_file_path)

    logging.info("Welcome!")
    logging.info(
        f"Modality: {modality}, Lang: {lang}, Modality: {modality}, Task: {task}"
    )

    logging.info(f"Output Json: {output_file_path}")


    # Loading data
    logging.info(f"Loading Data.")
    data = load_data(task=task, language=lang)
    input_data, references = data["inputs"][0:3], data["references"][0:3]

    # Loading Prompts
    logging.info(f"Loading Prompts.")
    prompt_dict = load_prompt(task=task, language=lang)

    # Loading Model
    logging.info(f"Loading Model.")
    model_instance, generate = load_model(model)


    logging.info(f"Starting Output Generation.")
    outputs = []

    for x, ref in tqdm(zip(input_data, references), desc="Generating Outputs", total=len(input_data)):
        out = {"ref": ref, "predicted": {}}
        for prompt_type, prompts in prompt_dict.items():
            out["predicted"][prompt_type] = {}
            for p in prompts[:1]:
                # text prompt
                text_prompt = {"prompt_modality": "text", "prompt": p["text"]}
                
                # female audio prompt
                f_p = p["female_rec"][0] if len(p["female_rec"]) >= 1 else None
                f_audio_prompt = {"prompt_modality": "audio", "prompt": f_p}

                # male audio prompt
                m_p = p["male_prompts"] if len(p["male_rec"]) >= 1 else None
                m_audio_prompt = {"prompt_modality": "audio", "prompt": m_p}

                # generate
                out["predicted"][prompt_type]["text"]  = generate(model_instance, text_prompt, x, modality)
                if f_p:
                    out["predicted"][prompt_type]["f_audio_prompt"]  = generate(model_instance, f_audio_prompt, x, modality)
                if m_p:
                    out["predicted"][prompt_type]["m_audio_prompt"]  = generate(model_instance, m_audio_prompt, x, modality)

        outputs.append(out)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    logging.info(f"Writing Outputs to file {output_file_path}.")
    logging.info("All done.")


if __name__ == "__main__":
    LANGS = ["alb", "cs", "de", "en", "es", "fr", "hu", "it", "lv", "nl", "pt", "ru", "sv"]
    MODALITIES = ["text", "audio"]
    TASKS = ["ACHAP", "ASR", "MT", "S2ST", "SLU", "SQA", "SSUM", "ST", "TSUM", "TTS"]
    MODELS = ["phi_multimodal", "qwen_omni"]

    parser = argparse.ArgumentParser(description="Process MCIF data.")

    parser.add_argument(
        "--lang", choices=LANGS, default=LANGS[0], help="Language to process"
    )
    parser.add_argument(
        "--modality", choices=MODALITIES, default=MODALITIES[0], help="Modality type"
    )
    parser.add_argument(
        "--task", choices=TASKS, default=TASKS[0], help="Task"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0], help="Model type")
    parser.add_argument(
        "--out_folder", default="generated_output", help="Output data folder path"
    )

    args = parser.parse_args()

    main(
        out_folder=args.out_folder,
        model=args.model,
        task=args.task,
        modality=args.modality,
        lang=args.lang,
    )
    # python main.py --lang de --modality audio --model phi_multimodal --task ASR --out_folder outputs