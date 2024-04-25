# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:48:58 2024

@author: Ridhwan Dewoprabowo
"""

from path_retriever import retrieve_path
from torch import nn

class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds




def main():
    entity = "Q177329"
    
if __name__ == '__main__':
    main()