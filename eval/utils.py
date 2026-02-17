from eval.asr import score_asr
from eval.achap import score_achap
from eval.mt import score_mt
from eval.s2st import score_s2st
from eval.sqa import score_sqa
from eval.ssum import score_ssum
from eval.st import score_st
from eval.tsum import score_tsum
from eval.tts import score_tts

def load_comet():
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da")).to("cuda")
    return model

def load_whisper():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )
    return pipe

def load_utmos_predictor():
    import torch
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).to("cuda")
    return predictor

def get_score_function(task):
    task = task.lower()
    eval_model = None
    if task == "asr": score_function = score_asr
    elif task == "achap": score_function = score_achap
    elif task == "mt": 
        score_function = score_mt
        eval_model = load_comet()
    elif task == "s2st": 
        score_function = score_s2st
        eval_model1 = load_whisper()
        eval_model2 = load_comet()
        eval_model3 = load_utmos_predictor()
        eval_model = [eval_model1, eval_model2, eval_model3]
        
    elif task == "sqa": score_function = score_sqa
    elif task == "ssum": score_function = score_ssum
    elif task == "st": 
        score_function = score_st
        eval_model = load_comet()
    elif task == "tsum": score_function = score_tsum
    elif task == "tts": 
        score_function = score_tts
        eval_model1 = load_whisper()
        eval_model2 = load_utmos_predictor()
        eval_model = [eval_model1, eval_model2]
    else:
        raise NotImplementedError()

    return score_function, eval_model