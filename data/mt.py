import os
def load_mt(language):
    raise NotImplementedError()

    # example in asr.py 

    if language == "":
        raise ValueError("_ is not supported.")
    
    # we do only en-x

    base_dir = "data_storage/task"
    os.makedirs(base_dir, exist_ok=True)

    audio_paths = [] # .wav should be stored in data_storage/task_name (loaded if already there, else download it)
    references = [] # strings

    return {"inputs" : audio_paths, "references": references}