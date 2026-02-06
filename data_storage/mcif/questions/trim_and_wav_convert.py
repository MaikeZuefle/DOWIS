import os
import sys
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def trim_silence_with_padding(audio, silence_thresh=-40, chunk_size=10, padding=500):
    """Trims silence from start and end of an AudioSegment, leaving padding."""
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if not nonsilent_ranges:
        return AudioSegment.silent(duration=100)

    start_trim = max(0, nonsilent_ranges[0][0] - padding)
    end_trim = min(len(audio), nonsilent_ranges[-1][1] + padding)
    return audio[start_trim:end_trim]


def process_audio_files(input_dir, output_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Supported formats
    valid_extensions = (".m4a", ".mp3", ".wav")

    # Get all files with valid extensions
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"No valid files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    for f in files:
        input_path = os.path.join(input_dir, f)

        # Create new filename (keep original name, change extension to .wav)
        file_name_no_ext = os.path.splitext(f)[0]
        new_filename = f"{file_name_no_ext}.wav"
        output_path = os.path.join(output_dir, new_filename)

        # Detect format for pydub
        current_ext = os.path.splitext(f)[1][1:].lower()

        try:
            # Load, Trim, and Export
            audio = AudioSegment.from_file(input_path, format=current_ext)
            trimmed_audio = trim_silence_with_padding(audio)
            trimmed_audio.export(output_path, format="wav")
            print(f"Success: {f} -> {new_filename}")
        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == "__main__":
    # Check if correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python trim_and_wav_convert.py <INPUT_DIR> <OUTPUT_DIR>")
    else:
        dir_in = sys.argv[1]
        dir_out = sys.argv[2]
        process_audio_files(dir_in, dir_out)