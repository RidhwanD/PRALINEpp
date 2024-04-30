# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:49:39 2024

@author: Ridhwan Dewoprabowo
"""

import os
import json
import argparse
import re
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
    pattern = r'^Q\d+$'
    paths = retrieve_all_paths(partition)
    entities = list(paths.keys())
    for key in paths:
        for path in paths[key]:
            for obj in path[1]:
                if (bool(re.match(pattern, obj)) and obj not in entities):
                    entities.append(obj)
            if bool(re.match(pattern, path[2])) and path[2] not in entities:
                entities.append(path[2])
    
    return entities

def retrieve_all_relations(partition):
    pattern = r'^P\d+$'
    paths = retrieve_all_paths(partition)
    relations = []
    for key in paths:
        for path in paths[key]:
            for objs in path[1]:
                obj = objs.split('-')
                # if (obj[0].startswith('P') and len(obj) > 1 and obj[0] not in relations):
                if bool(re.match(pattern, obj[0])) and obj[0] not in relations:
                    relations.append(obj[0])
    
    return relations

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