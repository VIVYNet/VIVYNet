# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# BERT Imports
from transformers import BertTokenizer

# Fairseq and SymphonyNet Imports
from symphonynet.src.fairseq.gen_utils import process_prime_midi, gen_one, get_trk_ins_map, get_note_seq, note_seq_to_midi_file, music_dict
from fairseq.models import FairseqLanguageModel


# Miscellaneous Import
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import inspect
import math
import os
import time


def main():
    src_input = input("Enter Text: ")
    tgt_input = [[2,2,2,1,0,0]] # Target input should always start at BOS


    """
    Process Text
    """
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    encoded = bert_tokenizer(
        src_input, 
        truncation=True,
        padding="max_length", 
        max_length=512
        )["input_ids"] # We are using only input_ids for now
    
    """
    Process Midi
    """
    MAX_POS_LEN = 4096
    PI_LEVEL = 2
    IGNORE_META_LOSS = 1
    RATIO = 4
    BPE = "_bpe" # or ""

    DATA_VOC_DIR=f"data/final/labels/vocabs/"
    DATA_BIN_DIR=f"data/final/labels/bin"
    BPE_DIR=f"symphonynet/data/bpe_res/"

    music_dict.load_vocabs_bpe(DATA_VOC_DIR, BPE_DIR if BPE == '_bpe' else None)


    """
    Model Initialization
    """
    CKPT_DIR = ""
    INFERENCE_DIR = "vivynet/inference"
    vivynet = FairseqLanguageModel.from_pretrained('.',
        checkpoint_file=CKPT_DIR,
        data_name_or_path=DATA_BIN_DIR,
        user_dir=INFERENCE_DIR)

    """
    Generation
    """
    while(True):
        try:
            generated, ins_logits = gen_one(vivynet, tgt_input, MIN_LEN = 1024)
            break
        except Exception as e:
            print(e)
            continue
    trk_ins_map = get_trk_ins_map(generated, ins_logits)
    note_seq = get_note_seq(generated, trk_ins_map)
    # timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime()) 
    # output_name = f'output_prime{max_measure_cnt}_chord{max_chord_measure_cnt}_{timestamp}.mid'
    # note_seq_to_midi_file(note_seq, output_name)

main()