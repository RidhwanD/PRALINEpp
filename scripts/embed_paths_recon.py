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
from pathlib import Path
from torch import nn
from context_utils import make_char_vocab, make_word_vocab
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from embedding_utils import get_window_segments
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerWordEmbeddings

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
parser.add_argument('--partition', default='val', choices=['train', 'val', 'test'], help='Partition')
parser.add_argument('--model', default='bert-base-uncased', help='Pretrained model')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# read paths
paths = {}
with open(f'{args.data_path}/{args.partition}/paths.json') as json_file:
    paths = json.load(json_file)
    
rel_attributes_dict = {}
with open(f'{args.data_path}/{args.partition}/relation_attributes.json') as json_file:
    rel_attributes_dict = json.load(json_file)
    
entity_attributes_dict = {}
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
        self.fc = nn.Linear(input_size, self.output_size).to(DEVICE)
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        # Pass through the linear layer
        x = self.fc(x)
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
        # print(word_embed_dim + char_embed_dim, hidden_dim, num_layers)
        # 1D convolutional network (CNN)
        self.conv1d = nn.Conv1d(2 * hidden_dim, num_classes, conv_filter_size)
        
        self.custom_layer = CustomLayer(output_size=768)
        
    def forward(self, word_char_indices):
        embeds = []
        for (text, word_index, char_index) in word_char_indices:
            # print(text)
            word_embeds = self.word_embeddings(word_index.to(DEVICE))
            char_embeds = self.char_embeddings(char_index.to(DEVICE))
            # print(word_embeds.shape)
            # print("------------------------------------------------------------------")
            # print(char_embeds.shape)
            # print("------------------------------------------------------------------")
            
            # Average pooling of char embedding
            word_embed_avg = torch.mean(char_embeds, dim=2, keepdim=True)
            # print(word_embed_avg.shape)
            # print("------------------------------------------------------------------")
            char_embeds = word_embed_avg.permute(0, 2, 1, 3)
            # print(char_embeds.shape)
            # print("------------------------------------------------------------------")
            
            # print(word_embeds.shape, char_embeds.shape)
            # Concatenate character embeddings and word embeddings
            concatenated_embeddings = torch.cat((char_embeds, word_embeds), dim=3)
            # print(concatenated_embeddings.shape)
            # print(concatenated_embeddings.shape)
            # print("------------------------------------------------------------------")
            
            lstm_output, _ = self.lstm(concatenated_embeddings.squeeze(1))
            # print(lstm_output.shape)
            # print("------------------------------------------------------------------")
            embeds.append(lstm_output)
        
        stacked_outputs = torch.cat(embeds, dim=1).permute(0, 2, 1)
        # print(stacked_outputs.shape)
        # print("------------------------------------------------------------------")
        
        cnn_output = self.conv1d(stacked_outputs)
        
        embedding = self.custom_layer(cnn_output)
        
        return embedding
    
class BERTAttributeEmbedding(nn.Module):
    def __init__(self, char_vocab_size, drop_out_rate, word_embed_dim, char_embed_dim,
                 hidden_dim, num_layers, num_classes, conv_filter_size):
        super(BERTAttributeEmbedding, self).__init__()

        # BERT model and tokenizer
        self.bert_model = DocumentPoolEmbeddings([TransformerWordEmbeddings(args.model, layers='-1', pooling_operation='mean')])

        # Character embeddings
        self.char_embeddings = CharEmbeddings(char_vocab_size, char_embed_dim, drop_out_rate)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(self.bert_model.embedding_length + char_embed_dim, hidden_dim, 
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        
        # 1D convolutional network (CNN)
        self.conv1d = nn.Conv1d(2 * hidden_dim, num_classes, conv_filter_size)
        
    def forward(self, word_char_indices):
        embeds = []
        for (text, word_index, char_index) in word_char_indices:
            # BERT word embeddings
            flair_sentence = Sentence(text.lower())
            self.bert_model.embed(flair_sentence)
            bert_output = flair_sentence.embedding
            
            # Character embeddings
            char_embeds = self.char_embeddings(char_index.to(DEVICE))
            
            # Average pooling of char embeddings
            word_embed_avg = torch.mean(char_embeds, dim=2, keepdim=True)
            char_embeds = word_embed_avg.permute(0, 2, 1, 3)
            word_embeds = bert_output.view(1, 1, char_embeds.shape[2], -1)
            
            print(word_embeds.shape, char_embeds.shape)
            # Concatenate BERT embeddings and character embeddings
            concatenated_embeddings = torch.cat((char_embeds, word_embeds), dim=3)
            print(concatenated_embeddings.shape)
            
            lstm_output, _ = self.lstm(concatenated_embeddings.squeeze(1))
            embeds.append(lstm_output)
        
        stacked_outputs = torch.cat(embeds, dim=1).permute(0, 2, 1)
        cnn_output = self.conv1d(stacked_outputs).unsqueeze(1)
        
        return cnn_output


## TODO: Make input for AttributeEmbedding a tensor so we can use cuda!!!

def embed_paths(p, window_size = 510, stride = 255):
    # Initialize BERT model and tokenizer
    bert_model_name = args.model
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, device=DEVICE)
    bert_model = BertModel.from_pretrained(bert_model_name).to(DEVICE)
    
    char_vocab = make_char_vocab(rel_attributes_dict)
    word_vocab = make_word_vocab(rel_attributes_dict)
    
    char_vocab_ent = make_char_vocab(entity_attributes_dict)
    word_vocab_ent = make_word_vocab(entity_attributes_dict)
    # model = BERTAttributeEmbedding(len(char_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size']).to(DEVICE)
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size']).to(DEVICE)
    
    # create embeddings
    HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_reconcoba_paths.h5'
    with h5py.File(HDF5_DIR, 'w') as h5f:
        for startpoint, pts in tqdm(paths.items()):
            embeddings = []
            for path in pts:
                concatenated_embedding = torch.empty(1,p['entity_embed_dim'],0).to(DEVICE)
                mid = path[1]
                assert type(mid) is list
                
                for i, m in enumerate(mid):
                    if m.startswith('P') and m.split('-')[0] in rel_attributes_dict:
                        # print(m.split('-')[0])
                        # print("==================================================================")
                        predicate = rel_attributes_dict[m.split('-')[0]]
                        # print(predicate)
                        # print("------------------------------------------------------------------")
                        pred_indices = get_attribute_indices(predicate, word_vocab, char_vocab)
                        pred_indices = [(text, word_indices.unsqueeze(0), char_indices.unsqueeze(0)) for (text, word_indices, char_indices) in pred_indices]
                        # print(pred_indices)
                        # print("------------------------------------------------------------------")
                        pred_embedding = model(pred_indices)
                        # print(pred_embedding.shape)
                        # print("------------------------------------------------------------------")
                        pred_embedding = pred_embedding.squeeze(0)
                        # print(pred_embedding.shape)
                        # print("------------------------------------------------------------------")
                        concatenated_embedding = torch.cat((concatenated_embedding, pred_embedding), dim=2)

                    elif m.startswith('Q') and m.split('-')[0] in entity_attributes_dict:
                        # print(m.split('-')[0])
                        entity = entity_attributes_dict[m]
                        # print(predicate)
                        ent_indices = get_attribute_indices(entity, word_vocab_ent, char_vocab_ent)
                        ent_indices = [(text, word_indices.unsqueeze(0), char_indices.unsqueeze(0)) for (text, word_indices, char_indices) in ent_indices]
                        ent_embedding = model(ent_indices).squeeze(0)
                        concatenated_embedding = torch.cat((concatenated_embedding, ent_embedding), dim=2)
                        
                    else:
                        continue
                
                # Feed embedding into BERT
                reshaped_tensor = concatenated_embedding.view(1, -1)
                # print('--------------------------------------------------------------')
                # print(reshaped_tensor.shape)
                # print((reshaped_tensor.size(1) - (window_size)))
                num_chunks = max(1,(reshaped_tensor.size(1) - (window_size)) // stride + 2)
                chunk_embeddings = []
                # print(num_chunks)

                # Process each chunk
                for i in range(num_chunks):
                    # Extract chunk from the tensor
                    chunk = reshaped_tensor[:, i * stride:i * stride + window_size]
                    print('============================================================')
                    print(chunk.shape, i * stride, i * stride + window_size)
                    print(chunk)
                    print('-----------------------"special_tokens_chunk"-------------------------------')
                    
                    if chunk.size(1) < window_size:
                        chunk = torch.nn.functional.pad(chunk, (0, (window_size ) - chunk.size(1)), mode='constant', value=0)
                    # Add special tokens [CLS] and [SEP]
                    special_tokens_chunk = torch.cat((torch.tensor([[101]]).to(DEVICE), chunk, torch.tensor([[102]]).to(DEVICE)), dim=1)
                    print(special_tokens_chunk)
                    print('-----------------------"tokens_chunk"------------------------')
                    
                    # Create attention mask
                    attention_mask_chunk = torch.ones_like(special_tokens_chunk)
                    attention_mask_chunk[special_tokens_chunk == 0] = 0
                
                    # Convert tensor to list of strings (tokenization)
                    tokens_chunk = tokenizer.convert_ids_to_tokens(special_tokens_chunk.squeeze().tolist())
                    print(tokens_chunk)
                    print('------------------------"input_ids_chunk"------------------------')
                    
                    # Convert tokens to IDs
                    input_ids_chunk = tokenizer.convert_tokens_to_ids(tokens_chunk)
                    print(input_ids_chunk)
                    print('-------------------------"input_ids_tensor_chunk"--------------------')
                    
                    # Convert input_ids to tensor
                    input_ids_tensor_chunk = torch.tensor([input_ids_chunk])
                    print(input_ids_tensor_chunk)
                    print('------------------------------------------------------------')
                    
                    # Get BERT embeddings for the chunk
                    with torch.no_grad():
                        outputs_chunk = bert_model(input_ids_tensor_chunk.to(DEVICE), attention_mask=attention_mask_chunk.to(DEVICE))
                    
                        # For simplicity, assume we're using the pooled output
                        pooled_output = outputs_chunk.pooler_output
                        
                        # print(pooled_output.shape)
                        # Append chunk embeddings to the list
                        chunk_embeddings.append(pooled_output)
                    
                # Aggregate segment embeddings to obtain the final representation
                
                final_embeddings = torch.mean(torch.stack(chunk_embeddings), dim=0)
                embeddings.append(final_embeddings.cpu().tolist())
                break
            
            if embeddings:
                # save values
                h5f.create_dataset(name=startpoint, data=np.vstack(embeddings), compression="gzip", compression_opts=9)

def embed_paths_v2(p, window_size = 510, stride = 255):
    # Initialize BERT model and tokenizer
    bert_model_name = args.model
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, device=DEVICE)
    bert_model = BertModel.from_pretrained(bert_model_name).to(DEVICE)
    
    char_vocab = make_char_vocab(rel_attributes_dict)
    word_vocab = make_word_vocab(rel_attributes_dict)
    
    char_vocab_ent = make_char_vocab(entity_attributes_dict)
    word_vocab_ent = make_word_vocab(entity_attributes_dict)
    # model = BERTAttributeEmbedding(len(char_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size']).to(DEVICE)
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size']).to(DEVICE)
    
    # create embeddings
    HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_reconcoba_paths.h5'
    with h5py.File(HDF5_DIR, 'w') as h5f:
        for startpoint, pts in tqdm(paths.items()):
            embeddings = []
            for path in pts:
                concatenated_embedding = torch.empty(1,768).to(DEVICE)
                mid = path[1]
                assert type(mid) is list
                # print("=============================================================")
                for i, m in enumerate(mid):
                    if m.startswith('P') and m.split('-')[0] in rel_attributes_dict:
                        # print(m.split('-')[0])
                        # print("==================================================================")
                        predicate = rel_attributes_dict[m.split('-')[0]]
                        # print(predicate)
                        # print("------------------------------------------------------------------")
                        pred_indices = get_attribute_indices(predicate, word_vocab, char_vocab)
                        pred_indices = [(text, word_indices.unsqueeze(0), char_indices.unsqueeze(0)) for (text, word_indices, char_indices) in pred_indices]
                        # print(pred_indices)
                        # print("------------------------------------------------------------------")
                        pred_embedding = model(pred_indices)
                        # print(m.split('-')[0])
                        # print(pred_embedding.shape)
                        # print("-------------------------------------------------------------")
                        # print(pred_embedding.shape)
                        # print("------------------------------------------------------------------")
                        # pred_embedding = pred_embedding.squeeze(0)
                        # print(pred_embedding.shape)
                        # print("------------------------------------------------------------------")
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
                # print(path)
                # print(concatenated_embedding.shape)
                # print("-------------------------------------------------------------")
                final_embeddings = torch.mean(concatenated_embedding, dim=0, keepdim=True).squeeze(0)
                # print(final_embeddings.shape)
                # print("-------------------------------------------------------------")
                embeddings.append(final_embeddings.cpu().tolist())
                # break
            
            # break
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
                indices = val_batch['indices']
                outputs = model(indices)
                # val_loss = criterion(outputs, val_batch['labels'])
                # total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{params["nb_epoch"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
def generate_embeddings(model, data, params):
    data_loader = DataLoader(data, batch_size=params['batch_size'], shuffle=False)
    
    model.eval()
    with torch.no_grad():
        embeddings = []
        for batch in data_loader:
            indices = batch['indices']
            outputs = model(indices)
            embeddings.append(outputs.embedding.detach().cpu().tolist())
            
def main():
    # Load model parameters
    with open('recon_params.json', 'r') as f:
        p = json.load(f)
        
    embed_paths_v2(p)
            
def main_training():
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
    model = AttributeEmbedding(len(char_vocab), len(word_vocab), p['drop_out_rate_ent'], p['word_embed_dim'], p['char_embed_dim'], p['hidden_dim_ent'], p['num_entEmb_layers'], p['entity_embed_dim'], p['conv_filter_size'])
    train_entity_embedding(model, train_indices, val_indices, p)
    
if __name__ == '__main__':
    main()