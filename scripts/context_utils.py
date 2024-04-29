# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:05:04 2024

Taken from https://github.com/ansonb/RECON
"""

from collections import OrderedDict
import numpy as np
import torch
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

def make_char_vocab(data):
    char_vocab = OrderedDict()
    char_vocab['<PAD>'] = 0
    char_vocab['<UNK>'] = 1
    char_idx = 2
    for entity in data.values():
        # Process description
        for char in entity['desc']:
            if char not in char_vocab:
                char_vocab[char] = char_idx
                char_idx += 1
        # Process aliases
        for alias in entity['aliases']:
            for char in alias:
                if char not in char_vocab:
                    char_vocab[char] = char_idx
                    char_idx += 1
        # Process labels
        label = entity['label']
        for char in label:
            if char not in char_vocab:
                char_vocab[char] = char_idx
                char_idx += 1
        # Process instances
        for instance in entity['instances']:
            for char in instance['label']:
                if char not in char_vocab:
                    char_vocab[char] = char_idx
                    char_idx += 1
    return char_vocab

def make_word_vocab(data):
    word_vocab = OrderedDict()
    word_vocab['<PAD>'] = 0
    word_vocab['<UNK>'] = 1
    word_idx = 2
    for entity in data.values():
        # Process description
        for word in entity['desc'].split():
            if word not in word_vocab:
                word_vocab[word] = word_idx
                word_idx += 1
        # Process aliases
        for alias in entity['aliases']:
            for word in alias.split():
                if word not in word_vocab:
                    word_vocab[word] = word_idx
                    word_idx += 1
        # Process labels
        label = entity['label'].split()
        for word in label:
            if word not in word_vocab:
                word_vocab[word] = word_idx
                word_idx += 1
        # Process instances
        for instance in entity['instances']:
            for word in instance['label'].split():
                if word not in word_vocab:
                    word_vocab[word] = word_idx
                    word_idx += 1
    return word_vocab

def get_unique_entities(data):
    unique_entities = set()
    unique_entities.add(-1)
    
    entity_surface_forms = [['ALL_ZERO']]

    for d in data:
      for entity in d['vertexSet']:
        unique_entities.add(entity['kbID'])
        entity_surface_forms.append([d['tokens'][tp] for tp in entity['tokenpositions']])

    return list(unique_entities), entity_surface_forms

def get_batch_unique_entities(entity_indices, entity_surface_forms):
    unique_entities = {-1: ['ALL_ZERO']}
    unique_entities_count = {-1: 0}
    for i in range(entity_indices.shape[0]):
      for j in range(entity_indices.shape[1]):
        for k in range(entity_indices.shape[2]):
          unique_entities[entity_indices[i,j,k]] = entity_surface_forms[i,j,k]
          unique_entities_count[entity_indices[i,j,k]] = unique_entities_count.get(entity_indices[i,j,k],0) + 1

    unique_entities_set = []
    unique_entities_surface_forms = []
    ent_occurrence = []
    ent_index = []
    for k, v in unique_entities.items():
      unique_entities_set.append(k)
      unique_entities_surface_forms.append(v)
      ent_occurrence.append(unique_entities_count[k])
      ent_index.append(k)
    max_entity_pos = np.argmax(ent_occurrence)
    max_occurred_ent = ent_index[max_entity_pos]
    max_occurred_ent_pos = unique_entities_set.index(max_occurred_ent)

    return np.array(unique_entities_set), unique_entities_surface_forms, max_occurred_ent_pos