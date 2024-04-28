# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:49:39 2024

@author: Ridhwan Dewoprabowo
"""

import os
import json
import argparse
from pathlib import Path

def retrieve_all_paths(partition):
    # set root path
    ROOT_PATH = Path(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
    parser.add_argument('--partition', default=partition, choices=['train', 'val', 'test'], help='Partition')
    args = parser.parse_args()
    
    # read paths
    paths = {}
    with open(f'{args.data_path}/{args.partition}/paths.json') as json_file:
        paths = json.load(json_file)
    
    return paths

def retrieve_all_entities(partition):
    paths = retrieve_all_paths(partition)
    return paths.keys()

def retrieve_all_attributes(partition):
    # set root path
    ROOT_PATH = Path(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
    parser.add_argument('--partition', default=partition, choices=['train', 'val', 'test'], help='Partition')
    args = parser.parse_args()
    
    # read paths
    contexts = {}
    with open(f'{args.data_path}/{args.partition}/entity_attributes.json') as json_file:
        contexts = json.load(json_file)
    
    return contexts

def main():
    pass
    
if __name__ == '__main__':
    main()