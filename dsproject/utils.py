""" This module contains utility functions for the project."""
import json
import os

from tqdm import tqdm

def read_files(data_path : str = "data/part1",
                language : str = 'en',
                is_file_name : bool =False) -> tuple:
    """
    Reads all .txt and .json files in a given directory.
    Stores files by their filetype and category.
    Returns a tuple of 4 lists.
    """
    text_cs_files = [] # Computer Scientists text files
    text_sculpt_files = [] # Sculptors text files
    json_cs_files = [] # Computer Scientists json files
    json_sculpt_files = [] # Sculptors json files

    text_cs_files_name = [] # Computer Scientists text files
    text_sculpt_files_name = [] # Sculptors text files
    json_cs_files_name = [] # Computer Scientists json files
    json_sculpt_files_name = [] # Sculptors json files

    # Read all files in the directory
    for file in tqdm(os.listdir(data_path), desc="Loading files"):
        if file.endswith(f"ComputerScientists_{language}.txt"):
            current_file_path = os.path.join(data_path, file)
            with open(current_file_path, 'r', encoding='utf-8') as f:
                text_cs_files.append(f.read())
            text_cs_files_name.append(file)
        if file.endswith(f"Sculptors_{language}.txt"):
            current_file_path = os.path.join(data_path, file)
            with open(current_file_path, 'r', encoding='utf-8') as f:
                text_sculpt_files.append(f.read())
            text_sculpt_files_name.append(file)
        if file.endswith(f"ComputerScientists_{language}.json"):
            current_file_path = os.path.join(data_path, file)
            with open(current_file_path, 'r', encoding='utf-8') as f:
                json_cs_files.append(json.load(f))
            json_cs_files_name.append(file)
        if file.endswith(f"Sculptors_{language}.json"):
            current_file_path = os.path.join(data_path, file)
            with open(current_file_path, 'r', encoding='utf-8') as f:
                json_sculpt_files.append(json.load(f))
            json_sculpt_files_name.append(file)
    
    if is_file_name: # Return the file names
        return (text_cs_files_name, text_sculpt_files_name, json_cs_files_name, json_sculpt_files_name)
    else:
        return (text_cs_files, text_sculpt_files, json_cs_files, json_sculpt_files)
