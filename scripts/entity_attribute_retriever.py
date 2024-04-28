# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:52:47 2024

@author: Ridhwan Dewoprabowo
"""

import time
import os
import json
from SPARQLWrapper import SPARQLWrapper, JSON
from path_retriever import retrieve_all_entities
from pathlib import Path

def get_attributes_from_Wikidata(entity):
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql")
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
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql")
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

def retrieve_JSONready_context(entity):
    attributes = get_attributes_from_Wikidata(entity)
    instanceof = get_instanseof_from_Wikidata(entity)
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
    
def build_entity_attribute_data():
    # May raises HTTPError: Too Many Requests
    partition = "test"
    entities = retrieve_all_entities(partition)
    dataset = {}
    for i, entity in enumerate(entities):
        dataset[entity] = retrieve_JSONready_context(entity)
    
    save_to_JSON(dataset, partition)
    
    return dataset

def save_to_JSON(dataset, partition):
    ROOT_PATH = Path(os.path.dirname(__file__))
    data_path = str(ROOT_PATH.parent) + '/data/final'
    
    with open(f"{data_path}/{partition}/entity_attribute.json", "w") as f: 
        json.dump(dataset, f, indent=4)
    

