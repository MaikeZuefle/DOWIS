def load_model():
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        
        attn_implementation="flash_attention_2",
        cache_dir="/home/mzuefle/.cache/huggingface/hub",
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def generate(model_processor, prompt, example, modality):
    from qwen_omni_utils import process_mm_info
    model, processor = model_processor


    USE_AUDIO_IN_VIDEO = False
    if modality == "text":
        user_conv_content = [{"type": "text", "text": f"{prompt}\n{example}\n"}]
    elif modality == "audio":
        user_conv_content = [{"type": "audio", "audio": str(example)}, {"type": "text", "text": prompt}]

    else:
        raise NotImplementedError()

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

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(**inputs, 
                                    thinker_return_dict_in_generate=True,
                                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                    return_audio=False,
                                    max_new_tokens=32768, 
                                    thinker_max_new_tokens=32768)
    
    response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

    # postprocess
    return response


