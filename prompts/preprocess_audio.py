import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

AUDIO_DIR = "audio_prompts/italian_female2"
LANGUAGE="it"
GENDER="female2"
IN_FOMRAT="m4a" # "mp3", "m4a" "wav"

tasks = ["ASR", "ST", "SQA", "SSUM", "SLU", "TTS", "S2ST", "MT", "TSUM", "LIPREAD", "ACHAP"]
prompt_types = ["basic", "basic", "formal", "formal", "informal", "informal", "detailed", "detailed", "short", "short"]


# Get all .m4a files
files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(f".{IN_FOMRAT}")]

# Sort files numerically
files.sort(key=lambda x: int(os.path.splitext(x)[0]))


def trim_silence_with_padding(audio, silence_thresh=-40, chunk_size=10, padding=500):
    """
    Trims silence from start and end of an AudioSegment, leaving padding.
    :param audio: AudioSegment
    :param silence_thresh: in dBFS, lower = more sensitive
    :param chunk_size: ms
    :param padding: ms to leave at start and end
    :return: trimmed AudioSegment
    """
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=chunk_size, silence_thresh=silence_thresh)

    if not nonsilent_ranges:
        # if the audio is silent, return a tiny silence
        return AudioSegment.silent(duration=100)

    # Calculate start and end with padding
    start_trim = max(0, nonsilent_ranges[0][0] - padding)
    end_trim = min(len(audio), nonsilent_ranges[-1][1] + padding)

    return audio[start_trim:end_trim]



# Convert each file to .wav
for idx, f in enumerate(files):
    task_idx = idx // 10  # Which task
    prompt_idx = idx % 10  # Which prompt in the task
    number = 1 if prompt_idx % 2 == 0 else 2  # 1 for first of pair, 2 for second

    task_name = tasks[task_idx]
    prompt_name = prompt_types[prompt_idx]

    new_filename = f"{LANGUAGE}_{GENDER}_{task_name}_{prompt_name}_{number}.wav".lower()
    input_path = os.path.join(AUDIO_DIR, f)
    output_path = os.path.join(AUDIO_DIR, new_filename)

    # Convert to WAV

    audio = AudioSegment.from_file(input_path, format=IN_FOMRAT)
    audio.export(output_path, format="wav")
    print(f"{f} -> {new_filename}")