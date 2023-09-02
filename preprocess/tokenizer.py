"""
File Name:      tokenizer.py

Authors:        Benjamin Herrera

Date Created:   8 MAR 2023

Description:    Script to sort and filter out a downloaded dataset of digital
                representation
"""

# Imports
from fairseq.data.indexed_dataset import (
    MMapIndexedDatasetBuilder,
    MMapIndexedDataset,
)
from utils.preprocess_midi import preprocess_midi_run
from utils.make_data import midi_binarize
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import numpy as np
import shutil
import torch
import json
import re
import os
import concurrent

# Constants
# DATASET_LOC = "/mnt/d/Projects/VIVY/Data/Good"
DATASET_LOC = "./data/Raw"
DATASET_INDEX = json.load(open(f"{DATASET_LOC}/index_en.json"))
DATA_TOK_LOC = "../data"
FINAL_LOC = "../data/final/features"

MIDI_TOK_LOC = f"{DATA_TOK_LOC}/midis"
TEXT_TOKEN_FILE = f"{DATA_TOK_LOC}/tokens/data.x"
MIDI_TOKEN_FILE = f"{DATA_TOK_LOC}/tokens/data.y"
MIDI_TOKEN_TEMP_FILE = f"{DATA_TOK_LOC}/tokens/temp.data.y"
MIDI_TOKEN_MAP = f"{DATA_TOK_LOC}/tokens/data_pointer.info"
MIDI_TOKEN_FILE_DATA = None
MIDI_FILE_TO_TOKEN_MAP = {}

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")  # Tokenizer

TRAIN_SPLIT = 0.6 


def build_map() -> None:
    """MIDI Token Map Builder Function
    
    Description:
        Because the MIDI tokenizer will not output the tokens in a desired
        order, we must read the corresponding map that comes along the token
        file to map the file to its tokenized representation

    Information:
        :return: None
        :rtype: None
    """
    
    # Open the file and build a mapping from it
    with open(MIDI_TOKEN_MAP) as f:
        for line_num, line in enumerate(f):
            line = line.rstrip()
            MIDI_FILE_TO_TOKEN_MAP[line] = line_num + 1


def transfer(item: str) -> None:
    """File Transfer Function
    
    Description:
        Transfer files from the dataset to a midis folder
    
    Information:
        :param item: Path to process for tokenizing
        :type item: str
        :return: None
        :rtype: None
    """
    
    # Get the absolute path of the midi file
    path = DATASET_LOC + "/" + item["directory"].split("./")[-1]
    if not os.path.isdir(path):
        return
    mid_files = os.listdir(path)

    if len(mid_files) == 0:
        return
    # Copy file
    for i in range(len(mid_files)):
        shutil.copy(path + mid_files[i], f"{MIDI_TOK_LOC}/{mid_files[i]}")


def tokenize(item: dict) -> None:
    """Tokenizer Function
    
    Description:
        Tokenizes the features and labels of the give datapoint. Places them
        into a .text and .music file for text and MIDI tokens, respectively.

    Information:
        :param item: Dictionary/document to process for tokenizing
        :type item: dict
        :return: None
        :rtype: None
    """
    
    #
    #   TEXT ENCODING
    #
    # print(item["title"])
    # Extract text and path to the data
    text = item["text"]
    path = DATASET_LOC + item["directory"].split(".")[-1]
    
    # Clean text
    text = text.strip()  # Strip text
    text = re.sub("'{3}[0-9]{0,3}'{3}", "", text)  # Remove all {'''#'''}
    text = re.sub("[0-9]", "", text)  # Remove all numbers
    text = re.sub(  # Remove special characters after a punctuation expression
        "(?<=[.,?!;)(\"'\\*&^%$#!-=+{}<>])(?:\s)*([.,?!;)(\"'\\*&^%$#!-=+{}<>])+",
        "",
        text,
    )
    text = re.sub(
        "[\[\]\(\)\:\{\}\|]", "", text
    )  # Remove [, ], (, ), {, }, and | characters
    text = re.sub("'{2,}", "", text)  # Remove redundatn ' characters
    text = re.sub(
        "<[^>]*>", "", text
    )  # Remove substrings surrounded by < and >
    text = re.sub(
        "\([^>]*\)", "", text
    )  # Remove substrings surrounded by ( and )
    text = re.sub(
        "\{[^>]*\}", "", text
    )  # Remove substrings surrounded by { and }
    text = re.sub(  # Remove lonely special characters
        "(?<![a-zA-Z0-9])[.,?!;)(\"'\\*&^%$#!-=+{}<>](?![a-zA-Z0-9])", "", text
    )
    text = re.sub(" +", " ", text)  # Remove redundant spaces
    text = (
        " " + text if re.match(r"^\W", text) else text
    )  # Add space to the front if line starts with a special character

    # Tokenize text
    encoded = TOKENIZER(text, truncation=True, max_length=512)["input_ids"]

    #
    #   MUSIC ENCODING
    #
    
    # Get the absolute path of the item's respective midi file
    path = DATASET_LOC + "/" + item["directory"].split("./")[-1]
    if not os.path.isdir(path):
        return
    if len(os.listdir(path)) == 0:
        return
    filepath = "./data/midis" + "/" + os.listdir(path)[0]

    # Exit if the tokenized MIDI file is not there
    if filepath not in MIDI_FILE_TO_TOKEN_MAP:
        print("not exist in db: " + item["title"])
        return

    line_num = MIDI_FILE_TO_TOKEN_MAP[filepath]  # Get line number of the file

    content = MIDI_TOKEN_FILE_DATA[line_num - 1]  # Getline token line content

    #
    #   WRITE STAGE
    #

    # Append tokenized text into a line in the targeted data directory
    with open(f"{TEXT_TOKEN_FILE}", "a+") as f:
        for i in encoded:
            f.write(str(i) + " ")        
        f.write("\n")
        
    # Append content to the finalized MIDI token file
    with open(f"{MIDI_TOKEN_FILE}", "a+") as f:
        f.write(f"{content}")


def text_binarize(train_ratio: float) -> None:
    """Text Binarization Method

    Description:
        Binarizes the text tokens into .idx and .bin files
        
    Information:
        :param train_ratio: the ratio between training and validation data
        :type train_ratio: float
        :return: None
        :rtype: None
    
    """
    
    # Variable declaration
    totalpiece = 0
    word_count = Counter()
    data = []
    
    # Send each line into the raw_data list
    with open(TEXT_TOKEN_FILE, "r") as f:
        for line in tqdm(f, desc="reading..."):
            totalpiece += 1
            data.append(line.strip())
        
    # count the words in each sentence and update the word_counts Counter
    for sentence in data:
        word_count.update(sentence.split())

    # sort the word counts in decreasing order of frequency
    sorted_word_counts = sorted(
        word_count.items(), key=lambda x: (-x[1], x[0]), reverse=False
    )

    # write the word counts to a text file in decreasing order of frequency
    with open(f"{FINAL_LOC}/dict.txt", "w") as f:
        for word, count in sorted_word_counts:
            f.write(f"{word} {count}\n")

    # Get the train size for the dataset
    train_size = int(totalpiece * train_ratio)

    # Binarize for training data
    train_ds = MMapIndexedDatasetBuilder(
        f"{FINAL_LOC}/train.bin", dtype=np.uint16
    )
    for item in tqdm(data[:train_size], desc="writing bin file (training)"):
        insert = [int(i) for i in item.split()]
        train_ds.add_item(torch.IntTensor(insert))
    train_ds.finalize(f"{FINAL_LOC}/train.idx")
    
    # Binarize for validation data
    valid_ds = MMapIndexedDatasetBuilder(
        f"{FINAL_LOC}/valid.bin", dtype=np.uint16
    )
    for item in tqdm(data[train_size:], desc="writing bin file (validation)"):
        insert = [int(i) for i in item.split()]
        valid_ds.add_item(torch.IntTensor(insert))
    valid_ds.finalize(f"{FINAL_LOC}/valid.idx")


# Main run thread
if __name__ == "__main__":
    # MultiThreading process to port files from the DB to the midis
    print("Copying Files Over...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        _ = list(
            tqdm(
                executor.map(transfer, DATASET_INDEX), total=len(DATASET_INDEX)
            )
        )

    # Run the MIDI preprocess script
    print("Preprocessing MIDIs...")
    preprocess_midi_run()
    
    # Build map
    print("Building Map...")
    build_map()
    
    # Read the files' lines
    print("Reading Token Temp File Content...")
    file = open(MIDI_TOKEN_TEMP_FILE) 
    MIDI_TOKEN_FILE_DATA = file.readlines()

    # print(list(MIDI_FILE_TO_TOKEN_MAP.keys())[:100])
    # exit()

    # MultiThreading process to tokenize data
    print("Synchronizing feature data to the label data...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        _ = list(
            tqdm(
                executor.map(tokenize, DATASET_INDEX), total=len(DATASET_INDEX)
            )
        )

    # Binarization step
    print("Binarization of the data")
    midi_binarize(TRAIN_SPLIT)
    text_binarize(TRAIN_SPLIT)

    # TEST The Dataset - Features
    feature_train_data = MMapIndexedDataset("../data/final/features/train")
    print("Feature Train Data Size:", len(feature_train_data))
    feature_valid_data = MMapIndexedDataset("../data/final/features/valid")
    print("Feature Valid Data Size:", len(feature_valid_data))
    print(
        "TOTAL FEATURE SIZE:", len(feature_train_data) + len(feature_valid_data)
    )

    # TEST The Dataset - Labels
    print()
    labels_train_data = MMapIndexedDataset("../data/final/labels/bin/train")
    a_count = 0
    for i in labels_train_data:
        if len(i) == 1:
            a_count += 1
    print("Label Valid Data Size:", a_count)
    labels_valid_data = MMapIndexedDataset("../data/final/labels/bin/valid")
    b_count = 0
    for i in labels_valid_data:
        if len(i) == 1:
            b_count += 1
    print("Label Valid Data Size:", b_count)
    print("TOTAL LABEL SIZE:", a_count + b_count)
