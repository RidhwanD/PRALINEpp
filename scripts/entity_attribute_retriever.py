# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:52:47 2024

@author: Ridhwan Dewoprabowo
"""

from SPARQLWrapper import SPARQLWrapper, JSON

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

def transform_to_JSONready(entity):
    attributes = get_attributes_from_Wikidata(entity)
    instanceof = get_instanseof_from_Wikidata(entity)
    result = {}
    result["desc"] = attributes["results"]["bindings"][0]["desc"]["value"]
    result["instances"] = []
    for re in instanceof["results"]["bindings"]:
        instance = {}
        instance["kbID"] = re["instance"]["value"].split("/")[-1]
        instance["label"] = re["label"]["value"]
        result["instances"].append(instance)
    result["aliases"] = [x.strip() for x in attributes["results"]["bindings"][0]["alt"]["value"].split(",")]
    result["label"] = attributes["results"]["bindings"][0]["label"]["value"]
        
    return result
    


