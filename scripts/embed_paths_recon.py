# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:48:58 2024

@author: Ridhwan Dewoprabowo
Taken from https://github.com/ansonb/RECON
"""

import json
import torch
import numpy as np
from torch import nn
from context_utils import make_char_vocab, make_word_vocab
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import OrderedDict

MAX_EDGES_PER_GRAPH = 72
MAX_NUM_NODES = 9

class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        """
        Initialize the CharEmbeddings module.

        Args:
            vocab_size (int): Size of the character vocabulary.
            embed_dim (int): Dimensionality of the character embeddings.
            drop_out_rate (float): Dropout rate for regularization.
        """
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        """
        Forward pass through the CharEmbeddings module.

        Args:
            words_seq (torch.Tensor): Tensor containing character indices for a sequence of words.

        Returns:
            torch.Tensor: Character embeddings after applying dropout.
        """
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds

class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_embeddings=None):
        """
        Initialize the WordEmbeddings module.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the word embeddings.
            pretrained_embeddings (torch.Tensor or None): Pretrained embeddings to initialize the embedding layer.
                If None, the embeddings are initialized randomly.
            padding_idx (int): Index used for padding token.
        """
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embeddings.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)

    def forward(self, word_indices):
        """
        Forward pass through the WordEmbeddings module.

        Args:
            word_indices (torch.Tensor): Tensor containing word indices.

        Returns:
            torch.Tensor: Word embeddings for the input word indices.
        """
        return self.embeddings(word_indices)


class EntityEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate, entity_embed_dim, conv_filter_size, entity_conv_filter_size, word_embed_dim, word_vocab, char_embed_dim, max_word_len_entity, char_vocab, char_feature_size):
        super(EntityEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate

        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim)
        #self.word_embeddings = word_embeddings
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, self.drop_rate)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
          bidirectional=bool(self.is_bidirectional))

        # self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len_entity + conv_filter_size - 1, max_word_len_entity + conv_filter_size - 1)

        self.conv1d_entity = nn.Conv1d(2*hidden_dim, entity_embed_dim, entity_conv_filter_size)
        self.max_pool_entity = nn.MaxPool1d(32, 32) # max batch len for context is 128

    def forward(self, words, chars, conv_mask):
        batch_size = words.shape[0]
        max_batch_len = words.shape[1]

        # words = words.view(words.shape[0]*words.shape[1],words.shape[2])
        # chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        print(words.shape, chars.shape, src_word_embeds.shape, char_feature.shape)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        outputs, hc = self.lstm(words_input)

        # h_drop = self.dropout(hc[0])
        h_n = hc[0].view(self.layers, 2, words.shape[0], self.hidden_dim)
        h_n = h_n[-1,:,:,:].squeeze() # (num_dir,batch,hidden)
        h_n = h_n.permute((1,0,2)) # (batch,num_dir,hidden)
        h_n = h_n.contiguous().view(h_n.shape[0],h_n.shape[1]*h_n.shape[2]) # (batch,num_dir*hidden)
        h_n_batch = h_n.view(batch_size,max_batch_len,h_n.shape[1])

        h_n_batch = h_n_batch.permute(0, 2, 1)
        conv_entity = self.conv1d_entity(h_n_batch)
        conv_mask = conv_mask.unsqueeze(dim=1)
        conv_mask = conv_mask.repeat(1,conv_entity.shape[1],1)
        conv_entity.data.masked_fill_(conv_mask.data, -float('inf'))

        max_pool_entity = nn.MaxPool1d(conv_entity.shape[-1], conv_entity.shape[-1])
        entity_embed = max_pool_entity(conv_entity)
        entity_embed = entity_embed.permute(0, 2, 1).squeeze(dim=1)
        
        return entity_embed

