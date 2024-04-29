# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:48:58 2024

@author: Ridhwan Dewoprabowo
Taken from https://github.com/ansonb/RECON
"""

import json
import torch
from torch import nn
from context_utils import make_char_vocab, make_word_vocab
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    
class EACEmbedding(nn.Module):
    """input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate, entity_embed_dim, conv_filter_size, entity_conv_filter_size, word_embed_dim, word_vocab, char_embed_dim, max_word_len_entity, char_vocab, char_feature_size"""
    
    def __init__(self, char_vocab_size, word_vocab_size, drop_out_rate, word_embed_dim, char_embed_dim,
                 hidden_dim, num_layers, num_classes, conv_filter_size):
        super(EACEmbedding, self).__init__()

        # Word embeddings
        self.word_embeddings = WordEmbeddings(word_vocab_size, word_embed_dim)

        # Character embeddings
        self.char_embeddings = CharEmbeddings(char_vocab_size, char_embed_dim, drop_out_rate)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(word_embed_dim + char_embed_dim, hidden_dim, 
                            num_layers=num_layers, bidirectional=True, batch_first=True)

        # 1D convolutional network (CNN)
        self.conv1d = nn.Conv1d(2 * hidden_dim, num_classes, conv_filter_size)

    def forward(self, word_indices, char_indices):
        
        # Word embeddings
        word_embeds = self.word_embeddings(word_indices)

        # Character embeddings
        char_embeds = self.char_embeddings(char_indices)

        # Concatenate word and character embeddings
        combined_embeds = torch.cat((word_embeds, char_embeds[:, :, -1, :]), dim=2)

        # Bidirectional LSTM encoder
        lstm_output, _ = self.lstm(combined_embeds)
        
        # Stack final outputs from BiLSTM
        stacked_outputs = torch.cat((lstm_output[:, -1, :lstm_output.size(2)//2], 
                                     lstm_output[:, 0, lstm_output.size(2)//2:]), dim=1)
        
        # 1D CNN
        cnn_output = self.conv1d(stacked_outputs.unsqueeze(2).repeat(1, 1, 3)).squeeze()

        return cnn_output

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert text data to numerical representations
def text_to_indices(text, word_vocab, char_vocab):
    word_indices = [word_vocab[word] for word in text.split() if word in word_vocab]
    # char_indices = [[char_vocab[char] for char in word if char in char_vocab] for word in text]
    char_indices = []
    for word in text.split():
        if word in word_vocab:
            word_indice = []
            for char in word:
                if char in char_vocab:
                    word_indice.append(char_vocab[char])
            char_indices.append(torch.tensor(word_indice))
    return word_indices, char_indices

def get_entity_text(entity_data):
    desc = entity_data.get('desc', '')
    aliases = entity_data.get('aliases', [])
    label = entity_data.get('label', '')
    instances = entity_data.get('instances', [])
    
    # Concatenate description, aliases, and label
    text = desc + " " + ' '.join(aliases) + " " + label
    
    # Add instance labels to the text
    for instance in instances:
        instance_label = instance.get('label', '')
        text += " " + instance_label
    
    return text

def get_all_indices(data, word_vocab, char_vocab):
    all_indices = []
    
    for entity_id, entity_data in data.items():
        text = get_entity_text(entity_data)
        res = text_to_indices(text, word_vocab, char_vocab)
        index = {}
        index['text'] = text
        index['words'] = torch.tensor(res[0])
        # index['chars'] = torch.tensor(res[1])
        index['chars'] = nn.utils.rnn.pad_sequence(res[1], batch_first=True, padding_value=0)
        all_indices.append(index)
    return all_indices

def train_entity_embedding(model, train_data, val_data, params):
    # Load train and val data
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Change this to the desired loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    
    for epoch in range(params['nb_epoch']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['nb_epoch']}"):
            # Extract batch data
            # print("TEXT", batch['text'])
            # print("WORDS", batch['words'])
            # print("CHARS", batch['chars'])
            words = batch['words']
            chars = batch['chars']

            # Forward pass
            optimizer.zero_grad()
            outputs = model(words, chars)

            # Compute loss
            # loss = criterion(outputs, batch['labels'])
            # total_loss += loss.item()

            # Backward pass and optimization
            # loss.backward()
            # optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for val_batch in val_loader:
                words = val_batch['words']
                chars = val_batch['chars']
                outputs = model(words, chars)
                # val_loss = criterion(outputs, val_batch['labels'])
                # total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{params["nb_epoch"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

'''
self, word_vocab_size, char_vocab_size, word_embed_dim, char_embed_dim,
             hidden_dim, num_layers, num_classes, conv_filter_size
'''

def main():
    # Load model parameters
    with open('recon_params.json', 'r') as f:
        p = json.load(f)
    
    # Load train and val data
    train_data = load_data('../data/final/train/entity_attributes.json')
    val_data = load_data('../data/final/val/entity_attributes.json')
    char_vocab = make_char_vocab(train_data)
    word_vocab = make_word_vocab(train_data)
    
    train_indices = get_all_indices(train_data, word_vocab, char_vocab)
    val_indices = get_all_indices(val_data, word_vocab, char_vocab)
    # Create and train the model
    model = EACEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size'])
    # model = EACEmbedding(p['char_embed_dim']+p['word_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['is_bidirectional_ent'], p['drop_out_rate_ent'], p['entity_embed_dim'], p['conv_filter_size'], p['entity_conv_filter_size'], p['word_embed_dim'], word_vocab, p['char_embed_dim'], p['max_char_len'], char_vocab, p['char_feature_size'])
    train_entity_embedding(model, train_indices, val_indices, p)
    
if __name__ == '__main__':
    main()