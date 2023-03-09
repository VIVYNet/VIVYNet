"""
File Name:      tokenizer.py

Authors:        Benjamin Herrera

Date Created:   8 MAR 2023

Description:    Script to sort and filter out a downloaded dataset of digital representation
"""

# Imports
from transformers import AutoTokenizer
from tqdm import tqdm
import concurrent.futures
import json

# Constants 
DATASET_LOC = "/mnt/d/Projects/VIVY/Data/Ready"
DATASET_INDEX = json.load(open(f"{DATASET_LOC}/index.json"))
DATA_TOK_LOC = "./data"
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")    # Tokenizer

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
    
    # Extract text and path to the data
    text = item["text"]
    path = DATASET_LOC + item["directory"].split(".")[-1]
    
    # Clean text
    text = text.replace("\n", "")
    
    # Tokenize text
    encoded = TOKENIZER(text)['input_ids']
    

# Main run thread
if __name__ == "__main__":
    
    # # MultiThreading process to tokenize features and labels
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     _ = list(tqdm(executor.map(tokenize, DATASET_INDEX), total=len(DATASET_INDEX)))
    
    for i in DATASET_INDEX:
        tokenize(i)