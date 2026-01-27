import os
def load_model():
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def generate(model_processor, prompt, example, modality, output_modality, out_wav):

    from qwen_omni_utils import process_mm_info
    import soundfile as sf

    # get (prompt-)modalities and model
    prompt_modality = prompt["prompt_modality"]
    orig_prompt = prompt["prompt"]
    model, processor = model_processor


    # prepare prompts
    if prompt_modality == "audio":
        prompt_dict = {"type": "audio", "audio": orig_prompt}

    elif prompt_modality == "text":
        prompt_dict = {"type": "text", "text": orig_prompt}

    # prepare inputs
    if modality == "audio":
        input_dict = {"type": "audio", "audio": example}
    elif modality == "text":
        input_dict = {"type": "text", "text": example}
        

    USE_AUDIO_IN_VIDEO = False
    RETURN_AUDIO = output_modality == "audio"

    user_conv_content = [input_dict, prompt_dict]

    conversations = [{"role": "user", "content": user_conv_content}]

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs, 
                                    thinker_return_dict_in_generate=True,
                                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                    return_audio=RETURN_AUDIO,
                                    max_new_tokens=32768)
    
    response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]

    if RETURN_AUDIO and audio is not None:
        sf.write(
            out_wav,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

    return response


