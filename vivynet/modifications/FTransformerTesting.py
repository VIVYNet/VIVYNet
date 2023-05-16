from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask
from transformers import BertModel

import logging, math
import os
import sys

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

model = TransformerEncoderBuilder.from_kwargs(
    n_layers = 12,
    n_heads = 16,
    query_dimensions= 512 // 16,
    value_dimensions= 512 // 16,
    feed_forward_dimensions = 4 * 512,
    activation = 'gelu',
    dropout = 0.1,
    attention_type = "causal-linear"
).get()

decoder_model = TransformerDecoderBuilder.from_kwargs(
                n_layers = 12,
                n_heads= 16,
                query_dimensions= 512 // 16,
                value_dimensions= 512 // 16,
                feed_forward_dimensions=4 * 512,
                activation='gelu',
                #final_normalization=True,
                dropout= 0.1,
                self_attention_type="causal-linear", 
                cross_attention_type="full", # Fully masked so that each domain can be merged
            ).get()

bert = BertModel.from_pretrained("bert-base-uncased")

print(model)

# for name, param in model.named_parameters():
#     print(f"{name}: {param}")

# for child in decoder_model.children():
#     for num, c in enumerate(child):
        # print(f"{num}: {c}")
        # print(c.self_attention.inner_attention)
        # for name, param in c.self_attention.named_parameters():
        #     param.requires_grad = True
        #     print(f"{num}./{name}: {param}")

# print(decoder_model.layers)
# # for child in decoder_model.children():
# #     print(child)
# for num, c in enumerate(decoder_model.layers):
#     # print(f"{num}: {c}")
#     for name, param in c.self_attention.named_parameters():
#         param.requires_grad = True
#         print(f"{num}./{name}: {param}")

# print()