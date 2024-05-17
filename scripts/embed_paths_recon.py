# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:48:58 2024

@author: Ridhwan Dewoprabowo
Taken from https://github.com/ansonb/RECON
"""

import argparse
import h5py
import json
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch import nn
from context_utils import make_char_vocab, make_word_vocab, make_char_vocab_list, make_word_vocab_list, preprocess_sentence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
parser.add_argument('--partition', default='test', choices=['train', 'val', 'test'], help='Partition')
parser.add_argument('--model', default='bert-base-uncased', help='Pretrained model')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# read paths
paths = {}
with open(f'{args.data_path}/{args.partition}/paths.json') as json_file:
    paths = json.load(json_file)
    
rel_attributes_dict = {}
with open(f'{args.data_path}/labeled_attributes.json') as json_file:
    rel_attributes_dict = json.load(json_file)
    
# entity_attributes_dict = {}
# with open(f'{args.data_path}/{args.partition}/entity_attributes.json') as json_file:
#     entity_attributes_dict = json.load(json_file)

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

class CustomLayer(nn.Module):
    def __init__(self, output_size):
        super(CustomLayer, self).__init__()
        self.output_size = output_size
        self.activation = nn.ReLU()  # You can use any activation function here

    def forward(self, x):
        # Calculate the input size dynamically
        input_size = x.size(1) * x.size(2)
        fc = nn.Linear(input_size, self.output_size) #.to(DEVICE)
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        # Pass through the linear layer
        x = fc(x)
        # Apply activation function
        x = self.activation(x)
        return x

class AttributeEmbedding(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, drop_out_rate, word_embed_dim, char_embed_dim,
                 hidden_dim, num_layers, num_classes, conv_filter_size):
        super(AttributeEmbedding, self).__init__()

        # Word embeddings
        self.word_embeddings = WordEmbeddings(word_vocab_size, word_embed_dim)

        # Character embeddings
        self.char_embeddings = CharEmbeddings(char_vocab_size, char_embed_dim, drop_out_rate)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(word_embed_dim + char_embed_dim, hidden_dim, 
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        
        # 1D convolutional network (CNN)
        self.conv1d = nn.Conv1d(2 * hidden_dim, num_classes, conv_filter_size)
        
        self.custom_layer = CustomLayer(output_size=768)
        
        # Final linear layer for classification
        self.fc = nn.Linear(768, 2)
        
    def forward(self, word_char_indices):
        embeds = []
        for (text, word_index, char_index) in word_char_indices:
            if not preprocess_sentence(text[0]):
                continue
            
            word_embeds = self.word_embeddings(word_index).squeeze(0)
            char_embeds = self.char_embeddings(char_index)
            
            # Average pooling of char embedding
            word_embed_avg = torch.mean(char_embeds.squeeze(0), dim=2, keepdim=True)
            char_embeds = word_embed_avg.permute(0, 2, 1, 3)
            
            concatenated_embeddings = torch.cat((char_embeds, word_embeds), dim=3)
            
            lstm_output, _ = self.lstm(concatenated_embeddings.squeeze(1))
            
            embeds.append(lstm_output)
        
        stacked_outputs = torch.cat(embeds, dim=1).permute(0, 2, 1)
        
        cnn_output = self.conv1d(stacked_outputs)
        
        embedding = self.custom_layer(cnn_output)
        
        output = self.fc(embedding)
        
        return output
        
    def forward_to_embedding(self, word_char_indices):
        embeds = []
        for (text, word_index, char_index) in word_char_indices:
            # print(text)
            if not preprocess_sentence(text):
                continue
            
            word_embeds = self.word_embeddings(word_index).squeeze(0)
            char_embeds = self.char_embeddings(char_index)
            
            # Average pooling of char embedding
            word_embed_avg = torch.mean(char_embeds.squeeze(0), dim=2, keepdim=True)
            char_embeds = word_embed_avg.permute(0, 2, 1, 3)
            
            concatenated_embeddings = torch.cat((char_embeds, word_embeds), dim=3)
            
            lstm_output, _ = self.lstm(concatenated_embeddings.squeeze(1))
            
            embeds.append(lstm_output)
        
        stacked_outputs = torch.cat(embeds, dim=1).permute(0, 2, 1)
        
        cnn_output = self.conv1d(stacked_outputs)
        
        embedding = self.custom_layer(cnn_output)
        
        return embedding

def embed_paths_trained(p):
    char_vocab = make_char_vocab(rel_attributes_dict)
    word_vocab = make_word_vocab(rel_attributes_dict)
    
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size']) #.to(DEVICE)
    model.load_state_dict(torch.load(f"saved_model_attr_{p['nb_epoch']}_pp.pth"))
    model.eval()
    
    # create embeddings
    HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_recon-trained-pp_paths.h5'
    with h5py.File(HDF5_DIR, 'w') as h5f:
        for startpoint, pts in tqdm(paths.items()):
            embeddings = []
            for path in pts:
                concatenated_embedding = torch.empty(1,768)
                mid = path[1]
                assert type(mid) is list
                for i, m in enumerate(mid):
                    if m.startswith('P') and m.split('-')[0] in rel_attributes_dict:
                        predicate = rel_attributes_dict[m.split('-')[0]]
                        pred_indices = get_attribute_indices(predicate, word_vocab, char_vocab)
                        pred_indices = [(text, word_indices.unsqueeze(0).unsqueeze(0), char_indices.unsqueeze(0).unsqueeze(0)) for (text, word_indices, char_indices) in pred_indices]
                        pred_embedding = model.forward_to_embedding(pred_indices)
                        concatenated_embedding = torch.cat((concatenated_embedding, pred_embedding), dim=0)
                    # elif m.startswith('Q') and m.split('-')[0] in entity_attributes_dict:
                        # print(m.split('-')[0])
                        # entity = entity_attributes_dict[m]
                        # print(predicate)
                        # ent_indices = get_attribute_indices(entity, word_vocab_ent, char_vocab_ent)
                        # ent_indices = [(text, word_indices.unsqueeze(0), char_indices.unsqueeze(0)) for (text, word_indices, char_indices) in ent_indices]
                        # ent_embedding = model(ent_indices).squeeze(0)
                        # concatenated_embedding = torch.cat((concatenated_embedding, ent_embedding), dim=2)
                        
                    else:
                        continue
                
                final_embeddings = torch.mean(concatenated_embedding, dim=0, keepdim=True).squeeze(0)
                embeddings.append(final_embeddings.cpu().tolist())
                
            if embeddings:
                # save values
                h5f.create_dataset(name=startpoint, data=np.vstack(embeddings), compression="gzip", compression_opts=9)


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Convert text data to numerical representations
def text_to_indices_tensor(text, word_vocab, char_vocab):
    # Tokenize the sentence into words
    words = preprocess_sentence(text)
    
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

def get_attribute_indices(data, word_vocab, char_vocab):
    indices = []
    # Process label
    label = data.get('label', '')
    if (label):
        indices.append(text_to_indices_tensor(label, word_vocab, char_vocab))
    
    # Process description
    desc = data.get('desc', '')
    if (desc):
        indices.append(text_to_indices_tensor(desc, word_vocab, char_vocab))
    
    # Process aliases
    aliases = data.get('aliases', [])
    for alias in aliases:
        if (alias):
           indices.append(text_to_indices_tensor(alias, word_vocab, char_vocab))
        
    # Process instances
    # instances = data.get('instances', [])
    # for instance in instances:
    #     inst_label = instance.get('label', '')
    #     if (inst_label):
    #         indices.append(text_to_indices_tensor(inst_label, word_vocab, char_vocab))
            
    return indices

def get_all_indices(data, word_vocab, char_vocab):
    all_indices = []
    
    for _, attr_data in data.items():
        indices = get_attribute_indices(attr_data, word_vocab, char_vocab)
        index = {'indices': indices}
        all_indices.append(index)
    return all_indices

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data, char_vocab, word_vocab):
        
        targets = []
        indices = []
        for predicate in data:
            # predicate = data[key]
            pred_indices = get_attribute_indices(predicate, word_vocab, char_vocab)
            pred_indices = [(text, word_indices.unsqueeze(0), char_indices.unsqueeze(0)) for (text, word_indices, char_indices) in pred_indices]
            indices.append(pred_indices)
            targets.append(predicate['cls'])
        
        self.data = indices
        self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train_attr_embedding(model, train_data, val_data, params):
    # Load train and val data
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Change this to the desired loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    for epoch in range(params['nb_epoch']):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)
        
        # Calculate average losses and accuracies
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{params['nb_epoch']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), f"saved_model_attr_{params['nb_epoch']}_pp.pth")
    print("Model saved successfully.")

def test_attr_embedding(model, test_data, params):
    # Set the model to evaluation mode
    model.eval()

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    # Define a criterion for computing the loss (if needed)
    criterion = nn.CrossEntropyLoss()

    # Initialize variables to track performance metrics
    test_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation during inference
    with torch.no_grad():
        # Iterate over batches in the test dataset
        for inputs, targets in tqdm(test_loader):
            # Forward pass to get model predictions
            outputs = model(inputs)

            # Compute the loss (if needed)
            if criterion:
                loss = criterion(outputs, targets)
                test_loss += loss.item()

            # Get the predicted class (index with highest probability)
            _, predicted = torch.max(outputs, 1)

            # Update total number of samples
            total += targets.size(0)

            # Update number of correctly classified samples
            correct += (predicted == targets).sum().item()

    # Calculate average test loss (if needed)
    if criterion:
        test_loss /= len(test_loader.dataset)

    # Calculate test accuracy
    test_acc = correct / total

    # Print test metrics
    if criterion:
        print(f"Test Loss: {test_loss:.4f}, ", end="")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc

def generate_embeddings(model, data, params):
    data_loader = DataLoader(data, batch_size=params['batch_size'], shuffle=False)
    
    model.eval()
    with torch.no_grad():
        embeddings = []
        for batch in data_loader:
            indices = batch['indices']
            outputs = model(indices)
            embeddings.append(outputs.embedding.detach().cpu().tolist())
            
def main_embed():
    # Load model parameters
    with open('recon_params.json', 'r') as f:
        p = json.load(f)
        
    embed_paths_trained(p)

def main_tr():
    # Load model parameters
    with open('recon_params.json', 'r') as f:
        p = json.load(f)
    
    data_dict = load_data('../data/final/labeled_attributes.json')
    data = list(data_dict.values())
    
    char_vocab = make_char_vocab_list(data)
    word_vocab = make_word_vocab_list(data)

    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Split train_data into train and validation sets (80% train, 20% validation)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    training(train_data, val_data, char_vocab, word_vocab, p)
    
    testing(test_data, char_vocab, word_vocab, p)

def training(train_data, val_data, char_vocab, word_vocab, p):
    train_dataset = CustomDataset(train_data, char_vocab, word_vocab)
    val_dataset = CustomDataset(val_data, char_vocab, word_vocab)

    # Create and train the model
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size'])
    train_attr_embedding(model, train_dataset, val_dataset, p)

def testing(test_data, char_vocab, word_vocab, p):
    test_dataset = CustomDataset(test_data, char_vocab, word_vocab)
    
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size'])
    model.load_state_dict(torch.load(f"saved_model_attr_{p['nb_epoch']}_pp.pth"))
    
    test_attr_embedding(model, test_dataset, p)
    
if __name__ == '__main__':
    main_embed()