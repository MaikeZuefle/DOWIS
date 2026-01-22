import xml.etree.ElementTree as ET
import logging
import os
import sys

def set_up_logging(output_file_path):
    log_file = output_file_path.replace(".xml", ".log")

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
