import logging as log
import os
from typing import Dict, Any

import numpy as np

from tcn.kodera29.data_loading.file_formats.FileFormat import FileFormat
from tcn.kodera29.data_loading.file_formats.SalehFileFormat import SalehFileFormat
from tcn.kodera29.data_loading.file_formats.MochuraFileFormat import MochuraFileFormat

preprocessed_data_folder = "preprocessed_data/"

def group_input_files_per_person(config: Dict[str, Any]) -> list[list[FileFormat]]:
    """
    Reads files from the ./data folder and groups the files that are measurements of the same person in a list.
    Returns a matrix, where a row represents a single person and the column is a single FileFormat
    """
    formats = [MochuraFileFormat, SalehFileFormat]

    # Parse the format of each input file
    processed_files = []
    for path, _, files in os.walk(config["data"]["data_dir"]):
        for file in files:
            file_path = path + '/' + file
            for file_format in formats:
                processed_file = file_format.process(file_path)
                if processed_file is not None:
                    processed_file.read_raw()
                    processed_files.append(processed_file)
                    break

    # Group files belonging to the same person in a list
    files_per_person = []
    for i in range(len(processed_files)):
        file = processed_files[i]
        # This file already belongs to a person which has been processed
        if file is None:
            continue

        person_files = [file]
        for j in range(i + 1, len(processed_files)):
            other_file = processed_files[j]
            if file.same_person(other_file):
                person_files.append(other_file)
                processed_files[j] = None  # Mark the file as processed

        processed_files[i] = None  # Mark the file as processed
        files_per_person.append(person_files)

    return files_per_person




def save_preprocessed_data(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> None:
    os.makedirs(preprocessed_data_folder, exist_ok=True)

    data_file_path = f"{preprocessed_data_folder}data_classes{config['network_args']['num_classes']}.npy"
    labels_file_path = f"{preprocessed_data_folder}labels_classes{config['network_args']['num_classes']}.npy"

    log.info(f"Saving preprocessed data to file {data_file_path} as a numpy ndarray.")
    log.info(f"Saving preprocessed labels to file {labels_file_path} as a numpy ndarray.")

    np.save(data_file_path, data, allow_pickle=True)
    np.save(labels_file_path, labels, allow_pickle=True)




def load_preprocessed_data(config: Dict[str, Any]) -> tuple:
    data_file_path = f"{preprocessed_data_folder}data_classes{config['network_args']['num_classes']}.npy"
    labels_file_path = f"{preprocessed_data_folder}labels_classes{config['network_args']['num_classes']}.npy"

    if not os.path.exists(data_file_path):
        log.warning(f"Unable to load preprocessed data from path {data_file_path}."
                    f" Maybe the data has not been preprocessed and saved yet?")
        return None, None

    log.info(f"Loading preprocessed data from {data_file_path}.")
    log.info(f"Loading preprocessed labels from {labels_file_path}.")

    data = np.load(data_file_path, allow_pickle=True)
    labels = np.load(labels_file_path, allow_pickle=True)

    return data, labels
