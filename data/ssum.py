import os
import gzip

import urllib.request
import xml.etree.ElementTree as ET

MCIF_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/resolve/main/"

def download_audio(base_dir):
    MCIF_LONGAUDIO_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/tree/main/MCIF_DATA/LONG_AUDIOS"
    with urllib.request.urlopen(MCIF_LONGAUDIO_URL) as r:
        html = r.read().decode("utf-8")

    files = []
    for part in html.split('"'):
        if part.endswith(".wav"):
            files.append(os.path.basename(part))

    print("Downloading audio...")
    for filename in sorted(set(files)):
        out_path = os.path.join(base_dir, filename)
        if os.path.exists(out_path):
            continue
        urllib.request.urlretrieve(
            f"{MCIF_URL+"MCIF_DATA/LONG_AUDIOS"}/{filename}", out_path)
    print("Done.")


def load_ssum(language):
    if language not in ["en", "it", "de"]:
        raise ValueError("Only English, Italian, and German languages are supported for SSUM.")

    base_dir = "data_storage/mcif"
    os.makedirs(base_dir, exist_ok=True)

    download_audio(base_dir)

    file_name = f"MCIF.long.{language}.ref.xml.gz"
    url = MCIF_URL + f"{file_name}?download=true"

    audio_paths = []; references = []

    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            context = ET.iterparse(gz, events=("end",))

            for event, elem in context:
                if elem.tag == "sample" and elem.attrib.get("task") == "SUM":
                    audio_paths.append(os.path.join(base_dir, elem.findtext("audio_path")))
                    references.append(elem.findtext("reference"))
                    elem.clear()
    return {"inputs": audio_paths, "references": references}