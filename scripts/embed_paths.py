import os
import json
import h5py
import torch
import flair
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, BertEmbeddings
from transformers import BertModel, BertTokenizer

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
parser.add_argument('--partition', default='test', choices=['train', 'val', 'test'], help='Partition')
parser.add_argument('--model', default='bert-base-uncased', help='Pretrained model')
args = parser.parse_args()

# read paths
paths = {}
with open(f'{args.data_path}/{args.partition}/paths.json') as json_file:
    paths = json.load(json_file)

# read labels dictionary for test set
labels_dict = {}
with open(f'{str(ROOT_PATH.parent)}/data/labels_dict.json') as json_file:
    labels_dict = json.load(json_file)
    
# read relation attribute dictionary for test set
rel_attributes_dict = {}
with open(f'{args.data_path}/{args.partition}/relation_attributes.json') as json_file:
    rel_attributes_dict = json.load(json_file)
    
# read entity attribute dictionary for test set
entity_attributes_dict = {}
with open(f'{args.data_path}/{args.partition}/entity_attributes.json') as json_file:
    entity_attributes_dict = json.load(json_file)

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
flair.device = DEVICE

def PRALINE():
    # load bert model
    pretrained_model = DocumentPoolEmbeddings([BertEmbeddings(args.model, layers='-1', pooling_operation='mean')])
    
    # create embeddings
    HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_paths.h5'
    with h5py.File(HDF5_DIR, 'w') as h5f:
        for startpoint, pts in tqdm(paths.items()):
            embeddings = []
            for path in pts:
                path_sentence = '[CLS] '
                mid = path[1]
                assert type(mid) is list
                for i, m in enumerate(mid):
                    if not m.startswith('P') or m.split('-')[0] not in labels_dict:
                        continue
    
                    predicate = m.split('-')[0]
                    sep_token = ' [SEP] ' if i > 0 else ' '
                    path_sentence = path_sentence + sep_token + labels_dict[predicate]
    
                flair_sentence = Sentence(path_sentence.lower())
                pretrained_model.embed(flair_sentence)
                embeddings.append(flair_sentence.embedding.detach().cpu().tolist())
    
            if embeddings:
                # save values
                h5f.create_dataset(name=startpoint, data=np.vstack(embeddings), compression="gzip", compression_opts=9)

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

def PRALINEpp(mode, window_size=512, stride=256):
    # Initialize BERT model and tokenizer
    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, device=DEVICE)
    bert_model = BertModel.from_pretrained(bert_model_name).to(DEVICE)
    
    print(f'Performing {mode} mode')
    # pretrained_model = DocumentPoolEmbeddings([bert_model])
    
    # create embeddings
    HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_augmented_{mode}_paths.h5'
    with h5py.File(HDF5_DIR, 'w') as h5f:
        for startpoint, pts in tqdm(paths.items()):
            embeddings = []
            for path in pts:
                path_sentence = '[CLS] '
                mid = path[1]
                assert type(mid) is list
                for i, m in enumerate(mid):
                    if m.startswith('P') and m.split('-')[0] in rel_attributes_dict:
                        predicate = rel_attributes_dict[m.split('-')[0]]
                        sep_token = ' [SEP] ' if i > 0 else ' '
                        token_with_attributes = f"{predicate['label']} [{', '.join(predicate['aliases'])}] ({predicate['desc']}())"
                        path_sentence = path_sentence + sep_token + token_with_attributes
                    elif m.startswith('Q') and m in entity_attributes_dict:
                        entity = entity_attributes_dict[m]
                        sep_token = ' [SEP] ' if i > 0 else ' '
                        token_with_attributes = f"{entity['label']} [{', '.join(entity['aliases'])}] ({entity['desc']}())"
                        path_sentence = path_sentence + sep_token + token_with_attributes
                    else:
                        continue

                if mode == 'truncate':
                    tokenized_inputs = tokenizer(path_sentence, truncation=True, max_length=512, return_tensors="pt")
                    with torch.no_grad():
                        outputs = bert_model(**tokenized_inputs)
                    # Get the contextualized embeddings from BERT
                    contextualized_embeddings = outputs.last_hidden_state
                    
                    # Aggregate contextualized embeddings to obtain sentence embedding (e.g., using mean pooling)
                    sentence_embedding = torch.mean(contextualized_embeddings, dim=1)     
                elif mode == 'sliding':
                    tokenized_inputs = tokenizer(path_sentence, return_tensors="pt")
                    window_segments = get_window_segments(tokenized_inputs, window_size, stride)
                    segment_embeddings = []
                    for window_input_ids, window_attention_masks in window_segments:
                        with torch.no_grad():
                            outputs = bert_model(input_ids=window_input_ids.to(DEVICE), attention_mask=window_attention_masks.to(DEVICE))
                        
                        # Get the output embeddings (e.g., pooled output or last layer hidden states)
                        # For simplicity, assume we're using the pooled output
                        pooled_output = outputs.pooler_output
                        
                        # Add the segment embeddings to the list
                        segment_embeddings.append(pooled_output)
                    
                    # Aggregate segment embeddings to obtain the final representation
                    sentence_embedding = torch.mean(torch.stack(segment_embeddings), dim=0)
    
                embeddings.append(sentence_embedding.cpu().tolist())
    
            if embeddings:
                # save values
                h5f.create_dataset(name=startpoint, data=np.vstack(embeddings), compression="gzip", compression_opts=9)



def main():
    model = "PRALINEpp"
    mode = 'sliding'                   # ['truncate', 'sliding']
    if model == "PRALINEpp":
        PRALINEpp(mode)
        
if __name__ == '__main__':
    main()
    
