# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:03:55 2024

Taken from https://github.com/ansonb/RECON
"""

# Embeddings and vocabulary utility methods

import numpy as np
import logging
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