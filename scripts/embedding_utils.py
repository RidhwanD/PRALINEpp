# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:03:55 2024

Taken from https://github.com/ansonb/RECON
"""

# Embeddings and vocabulary utility methods

import numpy as np
import logging
import torch
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.ERROR)

all_zeroes = "ALL_ZERO"
unknown = "_UNKNOWN"

special_tokens = {"&ndash;": "–",
                  "&mdash;": "—",
                  "@card@": "0"
                  }


def make_start_embedding(n, d):
    vector = []
    for i in range(n):
        for j in range(n):
            if(i != j):
                for k in range(n):
                    if(k == i):
                        vector += [1] * d + [0] * d
                    elif (k == j):
                        vector += [0] * d + [1] * d
                    else:
                        vector += [0] * d + [0] * d
    return np.array(vector, dtype=np.float).reshape(n * (n-1), n * 2 * d, 1)                

def get_head_indices(n, d, bs=50):
    indices = []
    for i in range(n):
        for j in range(n):
            if(i != j):
                for k in range(n):
                    if(k == i):
                        indices += [range(k * d * 2, k * d * 2 + d * 2)]
    return [indices] * bs

def get_tail_indices(n, d, bs=50):
    indices = []
    for i in range(n):
        for j in range(n):
            if(i != j):
                for k in range(n):
                    if(k == j):
                        indices += [range(k * d * 2, k * d * 2 + d * 2)]
    return [indices] * bs

def get_window_segments(tokenized_inputs, window_size, stride):
    # Get the tokenized input IDs and attention masks
    input_ids = tokenized_inputs['input_ids']
    attention_masks = tokenized_inputs['attention_mask']
    
    # Split the input sequence into overlapping windows
    window_segments = []
    for i in range(0, len(input_ids), stride):
        # Slice the input sequence to create a window
        window_input_ids = input_ids[:, i:i+window_size]
        window_attention_masks = attention_masks[:, i:i+window_size]
        
        # Add padding if the window is shorter than the maximum length
        if window_input_ids.shape[1] < window_size:
            padding_length = window_size - window_input_ids.shape[1]
            window_input_ids = torch.cat([window_input_ids, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
            window_attention_masks = torch.cat([window_attention_masks, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
        
        # Add the window to the list
        window_segments.append((window_input_ids, window_attention_masks))
    
    return window_segments