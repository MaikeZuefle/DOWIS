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
from utils import set_up_logging, TASK_MODALITY_MAPPER

# setting seed for reproducibilty
import random
set_seed(42)
random.seed(42)


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

def get_out_wav_path(output_modality, idx, wavs_folder, prompt_type):
    if output_modality == "audio":
        out_wav = f"{wavs_folder}/{idx}_{prompt_type}"
    text_prompt_wav = f"{out_wav}_text_prompt.wav" if output_modality == "audio" else None
    f_audio_prompt_wav = f"{out_wav}_f_audio_prompt.wav" if output_modality == "audio" else None
    m_audio_prompt_wav = f"{out_wav}_m_audio_prompt.wav" if output_modality == "audio" else None
    return text_prompt_wav, f_audio_prompt_wav, m_audio_prompt_wav

def main(out_folder, model, task, lang):

    # Setting output paths and inferring modalities
    output_file_path = f"{out_folder}/{model}/{task}/{lang}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    set_up_logging(output_file_path)
    modality, output_modality = TASK_MODALITY_MAPPER[task]["modality"], TASK_MODALITY_MAPPER[task]["output_modality"]
    wavs_folder = None
    if output_modality == "audio":
        wavs_folder = output_file_path.replace(".jsonl", "wavs")
        if not os.path.exists(wavs_folder):
            os.makedirs(wavs_folder)


    # Logging
    logging.info("Welcome!")
    logging.info(
        f"Modality: {modality}, Lang: {lang}, Task: {task}"
    )

    logging.info(f"Output Json: {output_file_path}")

    # Loading Data
    logging.info(f"Loading Data.")
    data = load_data(task=task, language=lang)
    input_data, references = data["inputs"], data["references"]

    # Loading Prompts
    logging.info(f"Loading Prompts.")
    prompt_dict = load_prompt(task=task, language=lang)

    # Loading Model
    logging.info(f"Loading Model.")
    model_instance, generate = load_model(model)

    # Starting Generation
    logging.info(f"Starting Output Generation.")

    f_out = open(output_file_path, "a", encoding="utf-8")
    
    for idx, (x, ref) in enumerate(
            tqdm(zip(input_data, references),
                desc="Generating Outputs",
                total=len(input_data))
        ):

        out = {"ref": ref, "predicted": {}}
        for prompt_type, prompts in prompt_dict.items():

            t_wav, fa_wav, ma_wav = get_out_wav_path(output_modality, idx, wavs_folder, prompt_type)
            out["predicted"][prompt_type] = {}

            # sample one out of two prompts in this category
            prompt_type_idx = random.randint(0, len(prompts) - 1)
            out["predicted"]["prompt_number"] = prompt_type_idx + 1 # either 1 or 2
            p = prompts[prompt_type_idx]
           
            # text prompt generation
            text_prompt = {"prompt_modality": "text", "prompt": p["text"]}
            out["predicted"][prompt_type]["text_prompt"]  = generate(model_instance, text_prompt, x, modality, output_modality, out_wav=t_wav)

            
            # sample one speaker (most tasks have only speaker per gender, but some have two)
            f_len, m_len = len(p["female_rec"]), len(p["male_rec"])
            num_audio_prompts = min(f_len, m_len) if f_len > 0 and m_len > 0 else max(f_len, m_len)
 
            if num_audio_prompts > 0:               
                speaker_idx = random.randint(0, num_audio_prompts - 1) if num_audio_prompts > 1 else 0
                out["predicted"]["spk_number"] = speaker_idx + 1 # either 1 or 2

                # audio prompts generation
                if  p["female_rec"]:
                    f_audio_prompt = {"prompt_modality": "audio", "prompt": p["female_rec"][speaker_idx]}
                    out["predicted"][prompt_type]["f_audio_prompt"]  = generate(model_instance, f_audio_prompt, x, modality, output_modality, out_wav=fa_wav)

                if p["male_rec"]:
                    m_audio_prompt = {"prompt_modality": "audio", "prompt": p["male_rec"][speaker_idx]}
                    out["predicted"][prompt_type]["m_audio_prompt"]  = generate(model_instance, m_audio_prompt, x, modality, output_modality, out_wav=ma_wav)

        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        f_out.flush()

    f_out.close()

    logging.info(f"Writing Outputs to file {output_file_path}.")
    logging.info("All done.")


if __name__ == "__main__":
    LANGS = ["cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "ru", "sq", "sv"]
    MODALITIES = ["text", "audio"]
    TASKS = ["ACHAP", "ASR", "MT", "S2ST", "SLU", "SQA", "SSUM", "ST", "TSUM", "TTS"]
    MODELS = ["phi_multimodal", "qwen_omni"]

    parser = argparse.ArgumentParser(description="Process MCIF data.")

    parser.add_argument(
        "--lang", choices=LANGS, default=LANGS[0], help="Language to process"
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
        lang=args.lang,
    )

    # Usage:
    # python main.py --lang es --model phi_multimodal --task MT --out_folder outputs_debug