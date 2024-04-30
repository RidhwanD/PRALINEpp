# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:52:47 2024

@author: Ridhwan Dewoprabowo
"""

import time
import os
import json
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from data_retriever import retrieve_all_entities, retrieve_all_relations
from pathlib import Path

def get_attributes_from_Wikidata(entity):
    user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql", user_agent)
    myid = "wd:"+entity
    sparqlwd.setQuery("""
                SELECT *
                WHERE {
                     SERVICE wikibase:label { bd:serviceParam wikibase:language "en".
                          """+myid+""" rdfs:label         ?label.
                          """+myid+""" skos:altLabel      ?alt.
                          """+myid+""" schema:description ?desc.
                      }
                }
                """
                      )
    sparqlwd.setReturnFormat(JSON)
    results = sparqlwd.query().convert()
    return results

def get_instanseof_from_Wikidata(entity):
    user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql", user_agent)
    myid = "wd:"+entity
    sparqlwd.setQuery("""
                SELECT *
                WHERE {
                     """+myid+""" wdt:P31 ?instance.
                     SERVICE wikibase:label { bd:serviceParam wikibase:language "en".
                         ?instance rdfs:label ?label.
                      }
                }
                """
                      )
    sparqlwd.setReturnFormat(JSON)
    results = sparqlwd.query().convert()
    return results

def retrieve_JSONready_context(rel_or_entity):
    attributes = get_attributes_from_Wikidata(rel_or_entity)
    instanceof = get_instanseof_from_Wikidata(rel_or_entity)
    result = {}
    try:
        result["desc"] = attributes["results"]["bindings"][0]["desc"]["value"]
    except KeyError:
        result["desc"] = ""
    result["instances"] = []
    for re in instanceof["results"]["bindings"]:
        instance = {}
        instance["kbID"] = re["instance"]["value"].split("/")[-1]
        instance["label"] = re["label"]["value"]
        result["instances"].append(instance)
    try:
        result["aliases"] = [x.strip() for x in attributes["results"]["bindings"][0]["alt"]["value"].split(",")]
    except KeyError:
        result["aliases"] = []
    try:
        result["label"] = attributes["results"]["bindings"][0]["label"]["value"]
    except KeyError:
        result["label"] = ""
        
    return result
    
def build_entity_attribute_data(partition):
    # May raises HTTPError: Too Many Requests. Adjust sleep time.
    entities = list(retrieve_all_entities(partition))
    
    dataset = {}
    for i, entity in enumerate(tqdm(entities)):
        try:
            dataset[entity] = retrieve_JSONready_context(entity)
        except:
            time.sleep(300)
            dataset[entity] = retrieve_JSONready_context(entity)
        # time.sleep(5)
    
    save_to_JSON(dataset, partition, "entity")
    
    return dataset

def build_relation_attribute_data(partition):
    # May raises HTTPError: Too Many Requests. Adjust sleep time.
    relations = list(retrieve_all_relations(partition))

    dataset = {}
    for i, rel in enumerate(tqdm(relations)):
        try:
            dataset[rel] = retrieve_JSONready_context(rel)
        except:
            time.sleep(30)
            dataset[rel] = retrieve_JSONready_context(rel)
        # time.sleep(5)
    
    save_to_JSON(dataset, partition, "relation")
    
    return dataset

def save_to_JSON(dataset, partition, obj):
    ROOT_PATH = Path(os.path.dirname(__file__))
    data_path = str(ROOT_PATH.parent) + '/data/final'
    
    with open(f"{data_path}/{partition}/{obj}_attributes.json", "w") as f: 
        json.dump(dataset, f, indent=4)
    
def main():
    # Create entity attributes dataset
    obj = "relation"
    for partition in ["test", "val", "train"]:
        print(f"Creating {partition} {obj} attributes dataset")
        build_entity_attribute_data(partition)
    
if __name__ == '__main__':
    main()
