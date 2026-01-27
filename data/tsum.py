import os
import gzip
import urllib.request
import xml.etree.ElementTree as ET

from datasets import load_dataset

MCIF_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/resolve/main/"
MCIF_VERSION = "1.2"

def load_tsum(language):
    if language not in ["en", "it", "de"]:
        raise ValueError("Only English, Italian, and German languages are supported for SQA.")

    base_dir = "data_storage/mcif"
    os.makedirs(base_dir, exist_ok=True)

    file_name = f"MCIF.long.{language}.ref.xml.gz"
    url = MCIF_URL + f"{file_name}?download=true"

    sum_samples = {}

    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            context = ET.iterparse(gz, events=("end",))
            for event, elem in context:
                if elem.tag == "sample" and elem.attrib.get("task") == "SUM":
                    sum_samples[f"{elem.attrib.get("id")}"] = {
                        "reference": elem.findtext("reference")
                    }
                    elem.clear()

    mcif_bench = load_dataset("FBK-MT/MCIF", "long_fixedprompt", split="test")

    for idx, entry in zip(mcif_bench["id"], mcif_bench[f"text"]):
        if idx in sum_samples:
            sum_samples[idx]["transcript"] = entry

    transcripts = []; references = []

    for idx in sum_samples.keys():
        transcripts.append(sum_samples[idx]["transcript"])
        references.append(sum_samples[idx]["reference"])

    return {"inputs": transcripts, "references": references}