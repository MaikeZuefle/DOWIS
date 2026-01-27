import xml.etree.ElementTree as ET
import logging
import os
import sys

TASK_MODALITY_MAPPER = {
    "ACHAP": {"modality": "audio", "output_modality": "text"},    
    "ASR": {"modality": "audio", "output_modality": "text"},      
    "MT": {"modality": "text", "output_modality": "text"},        
    "S2ST": {"modality": "audio", "output_modality": "audio"},    
    "SLU": {"modality": "audio", "output_modality": "text"},      
    "SQA": {"modality": "audio", "output_modality": "text"},       
    "SSUM": {"modality": "audio", "output_modality": "text"},      
    "ST": {"modality": "audio", "output_modality": "text"},       
    "TSUM": {"modality": "text", "output_modality": "text"},      
    "TTS": {"modality": "text", "output_modality": "audio"}       
}



def set_up_logging(output_file_path):
    log_file = output_file_path.replace(".jsonl", ".log")

    # Clear existing handlers if rerunning in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a unified format without milliseconds
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])



