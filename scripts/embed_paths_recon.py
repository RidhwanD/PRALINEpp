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

class EntityEmbedding(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, drop_out_rate, word_embed_dim, char_embed_dim,
                 hidden_dim, num_layers, num_classes, conv_filter_size):
        super(EntityEmbedding, self).__init__()

        # Word embeddings
        self.word_embeddings = WordEmbeddings(word_vocab_size, word_embed_dim)

        # Character embeddings
        self.char_embeddings = CharEmbeddings(char_vocab_size, char_embed_dim, drop_out_rate)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(word_embed_dim + char_embed_dim, hidden_dim, 
                            num_layers=num_layers, bidirectional=True, batch_first=True)

        # 1D convolutional network (CNN)
        self.conv1d = nn.Conv1d(2 * hidden_dim, num_classes, conv_filter_size)
        
    def forward(self, word_char_indices):
        embeds = []
        for (text, word_index, char_index) in word_char_indices:
            
            word_embeds = self.word_embeddings(word_index)
            char_embeds = self.char_embeddings(char_index)
            
            # Average pooling of char embedding
            word_embed_avg = torch.mean(char_embeds, dim=2, keepdim=True)
            
            char_embeds = word_embed_avg.permute(0, 2, 1, 3)
            
            # Concatenate character embeddings and word embeddings
            concatenated_embeddings = torch.cat((char_embeds, word_embeds), dim=2)
            
            lstm_output, _ = self.lstm(concatenated_embeddings.squeeze(1))
            embeds.append(lstm_output)
        
        stacked_outputs = torch.cat(embeds, dim=1)
        
        cnn_output = self.conv1d(stacked_outputs.permute(0, 2, 1)).unsqueeze(1)
        
        return cnn_output

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert text data to numerical representations
def text_to_indices_tensor(text, word_vocab, char_vocab):
    # Tokenize the sentence into words
    words = text.split()
    
    # Tokenize each word into characters and map to indices
    char_seq_indices = []
    max_word_length = 0
    for word in words:
        char_indices = [char_vocab.get(char, char_vocab['<PAD>']) for char in word]
        char_seq_indices.append(char_indices)
        max_word_length = max(max_word_length, len(char_indices))
    
    # Padding
    padded_char_seq_indices = [seq + [char_vocab['<PAD>']] * (max_word_length - len(seq)) for seq in char_seq_indices]
    word_seq_indices = [[word_vocab[word] for word in words]]  # Word indices
    
    # Convert to PyTorch tensor
    words_seq_tensor = torch.tensor(word_seq_indices)
    chars_seq_tensor = torch.tensor(padded_char_seq_indices)
    
    return text, words_seq_tensor, chars_seq_tensor

def get_all_indices(data, word_vocab, char_vocab):
    all_indices = []
    
    for entity_id, entity_data in data.items():
        entity_indices = []
        # Process label
        label = entity_data.get('label', '')
        if (label):
            entity_indices.append(text_to_indices_tensor(label, word_vocab, char_vocab))
        
        # Process description
        desc = entity_data.get('desc', '')
        if (desc):
            entity_indices.append(text_to_indices_tensor(desc, word_vocab, char_vocab))
        
        # Process aliases
        aliases = entity_data.get('aliases', [])
        for alias in aliases:
            if (alias):
               entity_indices.append(text_to_indices_tensor(alias, word_vocab, char_vocab))
            
        # Process aliases
        instances = entity_data.get('instances', [])
        for instance in instances:
            inst_label = instance.get('label', '')
            if (inst_label):
                entity_indices.append(text_to_indices_tensor(inst_label, word_vocab, char_vocab))
        index = {'indices': entity_indices}
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
            indices = batch['indices']
            # Forward pass
            optimizer.zero_grad()
            outputs = model(indices)

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
                indices = batch['indices']
                outputs = model(indices)
                # val_loss = criterion(outputs, val_batch['labels'])
                # total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{params["nb_epoch"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        

def main():
    # Load model parameters
    with open('recon_params.json', 'r') as f:
        p = json.load(f)
    
    # Load train and val data
    train_data = load_data('../data/final/train/entity_attributes.json')
    val_data = load_data('../data/final/val/entity_attributes.json')
    test_data = load_data('../data/final/test/entity_attributes.json')
    char_vocab = make_char_vocab({**train_data, **val_data, **test_data})
    word_vocab = make_word_vocab({**train_data, **val_data, **test_data})
    
    train_indices = get_all_indices(train_data, word_vocab, char_vocab)
    val_indices = get_all_indices(val_data, word_vocab, char_vocab)
    # Create and train the model
    model = EntityEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size'])
    train_entity_embedding(model, train_indices, val_indices, p)
    
if __name__ == '__main__':
    main()