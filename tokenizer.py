"""
File Name:      tokenizer.py

Authors:        Benjamin Herrera

Date Created:   8 MAR 2023

Description:    Script to sort and filter out a downloaded dataset of digital representation
"""

# Imports
from preprocess_midi import preprocess_midi_run
from transformers import AutoTokenizer
from tqdm import tqdm
import concurrent.futures
import shutil
import json
import re
import os

# Constants 
DATASET_LOC = "/mnt/d/Projects/VIVY/Data/Ready"
DATASET_INDEX = json.load(open(f"{DATASET_LOC}/index.json"))
DATA_TOK_LOC = "/home/blherre4/VIVY/VIVYNet/data"

MIDI_TOK_LOC = f"{DATA_TOK_LOC}/midis"
TEXT_TOKEN_FILE = f"{DATA_TOK_LOC}/tokens/data.x"
MIDI_TOKEN_FILE = f"{DATA_TOK_LOC}/tokens/data.y"
MIDI_TOKEN_TEMP_FILE = f"{DATA_TOK_LOC}/tokens/temp.data.y"
MIDI_TOKEN_FILE_DATA = None
MIDI_TOKEN_MAP = f"{DATA_TOK_LOC}/tokens/data_pointer.info"
MIDI_FILE_TO_TOKEN_MAP = {}

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")    # Tokenizer

def build_map() -> None:
    """MIDI Token Map Builder Function
    
    Description:
        Because the MIDI tokenizer will not output the tokens in a desired order, 
        we must read the corresponding map that comes along the token file to map
        the file to its tokenized representation
    
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
    mid_files = os.listdir(path)
    
    # Copy file
    shutil.copy(
        path + mid_files[0], 
        f"{MIDI_TOK_LOC}/{mid_files[0]}"
    )

def tokenize(item: dict) -> None:
    """Tokenizer Function
    
    Description:
        Tokenizes the features and labels of the give datapoint. Places them into a .text
        and .music file for text and MIDI tokens, respectively.
    
    Information:
        :param item: Dictionary/document to process for tokenizing
        :type item: dict
        :return: None
        :rtype: None
    """
    
    #
    #   TEXT ENCODING
    #
    
    # Extract text and path to the data
    text = item["text"]
    path = DATASET_LOC + item["directory"].split(".")[-1]
    
    # Clean text
    text = text.strip()                                     # Strip text
    text = re.sub("'{3}[0-9]{0,3}'{3}", '', text)           # Remove all {'''#'''}
    text = re.sub("[0-9]", "", text)                        # Remove all numbers
    text = re.sub(                                          # Remove special characters after a punctuation expression 
        "(?<=[.,?!;)(\"'\\*&^%$#!-=+{}<>])(?:\s)*([.,?!;)(\"'\\*&^%$#!-=+{}<>])+", 
        "", 
        text
    )
    text = re.sub("[\[\]\(\)\:\{\}\|]", "", text)           # Remove [, ], (, ), {, }, and | characters
    text = re.sub("'{2,}", "", text)                        # Remove redundatn ' characters
    text = re.sub("<[^>]*>", "", text)                      # Remove substrings surrounded by < and >
    text = re.sub("\([^>]*\)", "", text)                    # Remove substrings surrounded by ( and )
    text = re.sub("\{[^>]*\}", "", text)                    # Remove substrings surrounded by { and }
    text = re.sub(                                          # Remove lonely special characters
        "(?<![a-zA-Z0-9])[.,?!;)(\"'\\*&^%$#!-=+{}<>](?![a-zA-Z0-9])",
        "",
        text
    )
    text = re.sub(" +", " ", text)                          # Remove redundant spaces
    text = " " + text if re.match(r"^\W", text) else text   # Add space to the front if line starts with a special character
    
    # Tokenize text
    encoded = TOKENIZER(text, truncation=True, max_length=512)['input_ids']
    
    #
    #   MUSIC ENCODING
    #
    
    # Get the absolute path of the item's respective midi file
    path = DATASET_LOC + "/" + item["directory"].split("./")[-1]
    filepath = MIDI_TOK_LOC + "/" + os.listdir(path)[0]
    
    # Exit if the tokenized MIDI file is not there
    if filepath not in MIDI_FILE_TO_TOKEN_MAP:
        return
    
    line_num = MIDI_FILE_TO_TOKEN_MAP[filepath]     # Get line number of the file
    
    content = MIDI_TOKEN_FILE_DATA[line_num - 1]    # Getline token line content
    
    #
    #   WRITE STAGE
    #
    
    # Append tokenized text into a line in the targeted data directory
    with open(f"{TEXT_TOKEN_FILE}", "a") as f:
        for i in encoded:
            f.write(str(i) + " ")        
        f.write("\n")
    
    # Append content to the finalized MIDI token file
    with open(f"{MIDI_TOKEN_FILE}", "a") as f:
        f.write(f"{content}")
    
# Main run thread
if __name__ == "__main__":
    
    # # MultiThreading process to port files from the DB to the midis
    # print("Copying Files Over...")
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     _ = list(tqdm(executor.map(transfer, DATASET_INDEX), total=len(DATASET_INDEX)))
    
    # # Run the MIDI preprocess script
    # print("Preprocessing MIDIs...")
    # preprocess_midi_run()
    
    # Build map
    print("Building Map...")
    build_map()
    
    # Read the files' lines
    print("Reading Token Temp File Content...")
    file = open(MIDI_TOKEN_TEMP_FILE) 
    MIDI_TOKEN_FILE_DATA = file.readlines()
    
    # MultiThreading process to tokenize data
    print("Synchronizing feature data to the label data...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        _ = list(tqdm(executor.map(tokenize, DATASET_INDEX), total=len(DATASET_INDEX)))