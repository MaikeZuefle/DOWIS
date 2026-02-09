import os
import gzip
from huggingface_hub import snapshot_download

import urllib.request
import xml.etree.ElementTree as ET

MCIF_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/resolve/main/"

def download_audio(base_dir):
    print("Downloading audio...")
    snapshot_download(
        repo_id="FBK-MT/MCIF",
        repo_type="dataset",
        allow_patterns="MCIF_DATA/LONG_AUDIOS/*.wav",
        local_dir=base_dir,
        local_dir_use_symlinks=False
    )
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
                    audio_paths.append(
                        os.path.join(
                            base_dir, "MCIF_DATA/LONG_AUDIOS/", elem.findtext("audio_path"))
                    )
                    references.append(elem.findtext("reference"))
                    elem.clear()
    return {"inputs": audio_paths, "references": references}