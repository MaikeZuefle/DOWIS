import os
from datasets import load_dataset
from data.ssum import download_audio
import gzip

import urllib.request
import xml.etree.ElementTree as ET


MCIF_URL = "https://huggingface.co/datasets/FBK-MT/MCIF/resolve/main/"


def load_sqa(language):
    if language not in ["en", "it", "de"]:
        raise ValueError("Only English, Italian, and German languages are supported for SQA.")

    base_dir = "data_storage/mcif"
    os.makedirs(base_dir, exist_ok=True)

    download_audio(base_dir)

    mcif_bench = load_dataset("FBK-MT/MCIF", "long_fixedprompt", split="test")

    qa_starting_prompt = {
        "en": "Answer the following question concisely given the English content: ",
        "it": "Rispondi in modo conciso alla seguente domanda dato il contenuto inglese: ",
        "de": "Beantworte die folgende Frage kurz und bündig basierend auf dem englischen Inhalt: "
    }

    file_name = f"MCIF.long.{language}.ref.xml.gz"
    url = MCIF_URL + f"{file_name}?download=true"

    sqa_samples = {}

    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            context = ET.iterparse(gz, events=("end",))

            for event, elem in context:
                if elem.tag == "sample" and elem.attrib.get("task") == "SUM":
                    sqa_samples[elem.attrib.get("id")] = {
                        "audio_path": elem.findtext("audio_path"),
                        "reference": elem.findtext("reference")
                    }
                    elem.clear()

    for idx, entry in enumerate(mcif_bench):
        if idx in sqa_samples:
            question = entry[f"prompt_{language}"].replace(qa_starting_prompt[language], "")
            sqa_samples[idx]["question"] = question

    audio_paths = []; references = []; questions = []

    for idx in sqa_samples.keys():
        audio_paths.append(sqa_samples[idx]["audio_path"])
        references.append(sqa_samples[idx]["reference"])
        questions.append(sqa_samples[idx]["question"])

    return {"inputs": audio_paths, "references": references, "additional_inputs": questions}