# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:05:04 2024

Taken from https://github.com/ansonb/RECON
"""

from nltk import word_tokenize
from nltk import sent_tokenize
from collections import OrderedDict
import numpy as np
import torch
import embedding_utils
import json
import datetime

all_zeroes = "ALL_ZERO"
unknown = "_UNKNOWN"
MAX_CTX_SENT_LEN = 32
MAX_NUM_CONTEXTS = 32
CUDA = torch.cuda.is_available()

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
          return obj.tolist()
        elif isinstance(obj, datetime.datetime):
          return str(obj)
        elif isinstance(obj, np.bool_):
          return bool(obj)
        else:
            return json.JSONEncoder.default(self, obj)

def make_start_entity_embeddings(entity_embeddings, entity_pos_indices, unique_entities, embedding_dim, max_occurred_entity_in_batch_pos, start_embedding_template, max_num_nodes=9):
    # import time
    # s1=time.time()
    # import pdb; pdb.set_trace()
    if torch.cuda.is_available():
      vector = torch.ones((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim*max_num_nodes,1)).cuda()
    else:
      vector = torch.ones((entity_pos_indices.shape[0],entity_pos_indices.shape[1],2*embedding_dim*max_num_nodes,1))
    
    max_occurring_ent_emb_reshaped = entity_embeddings[max_occurred_entity_in_batch_pos].unsqueeze(-1).repeat([2*max_num_nodes,1])
    vector = vector*max_occurring_ent_emb_reshaped

    # s2=time.time()
    # print('t3',s2-s1)
    # import pdb; pdb.set_trace()
    for idx in range(entity_pos_indices.shape[0]):
      # for idx2 in range(entity_pos_indices.shape[1]):
      #     vector[idx,idx2,2*(idx2//max_num_nodes)*embedding_dim:(2*(idx2//max_num_nodes)+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,0]]
      #     vector[idx,idx2,(2*(idx2//max_num_nodes)+1 + 2*(idx2%max_num_nodes)+1)*embedding_dim:2*(idx2%max_num_nodes+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,1]]
      idx2 = 0
      count = 0
      for i in range(max_num_nodes):
          for j in range(max_num_nodes):
            if i!=j:
                  for k in range(max_num_nodes):
                      if(k == i):
                          if entity_pos_indices[idx,idx2,0]!=max_occurred_entity_in_batch_pos:
                            vector[idx,idx2,2*i*embedding_dim:(2*i+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,0]]
                          count += 1
                      elif (k == j):
                          if entity_pos_indices[idx,idx2,1]!=max_occurred_entity_in_batch_pos:
                            vector[idx,idx2,(2*j+1)*embedding_dim:2*(j+1)*embedding_dim,0] = entity_embeddings[entity_pos_indices[idx,idx2,1]]
                          count += 1
                      if count == 2:
                        count = 0
                        idx2 += 1
      
    vector = vector*start_embedding_template

    return vector      