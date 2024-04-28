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

