import os
import gzip
import json

import urllib.request
import xml.etree.ElementTree as ET

MCIF_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/resolve/main/"

def download_audio(base_dir):
    api_url = "https://huggingface.co/api/datasets/FBK-MT/MCIF/tree/main"

    with urllib.request.urlopen(api_url) as response:
        files = json.load(response)

    wav_files = [f["path"] for f in files if f["path"].endswith(".wav")]

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        out_path = os.path.join(base_dir, filename)

        if os.path.exists(out_path):
            continue

        download_url = MCIF_URL + file_path
        urllib.request.urlretrieve(download_url, out_path)


def load_ssum(language):
    if language not in ["en", "it", "de"]:
        raise ValueError("Only English, Italian, and German languages are supported for SQA.")

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
                    audio_paths.append(elem.findtext("audio_path"))
                    references.append(elem.findtext("reference"))
                    elem.clear()
    return {"inputs": audio_paths, "references": references}