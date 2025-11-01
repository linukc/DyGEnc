from typing import List

import networkx as nx
from pydantic import BaseModel


class Triplet(BaseModel):
    source: str
    edge: str
    target: str

class Target(BaseModel):
    noun: str
    adjective: str

class TripletGPT(BaseModel):
    source: str
    edge: str
    target: Target

class TripletExtraction(BaseModel):
    triplets: List[Triplet]

class TripletExtractionGPT(BaseModel):
    triplets: List[TripletGPT]

def triplets2graph_gpt(triplets):
    """
    Converts a list of triplets into a directed graph.

    Args:
        triplets (TripletExtraction): An object containing extracted triplets.

    Returns:
        networkx.DiGraph: A directed graph representing the triplets.
    """
    G = nx.DiGraph()
    for triplet in triplets.triplets:
        G.add_node(triplet.source.strip(), label=triplet.source.strip())
        G.add_node(f"{triplet.target.adjective.strip()} {triplet.target.noun.strip()}",
                   label=f"{triplet.target.adjective.strip()} {triplet.target.noun.strip()}")
        G.add_edge(triplet.source.strip(), f"{triplet.target.adjective.strip()} {triplet.target.noun.strip()}", label=triplet.edge.strip())
    return G

def triplets2graph(triplets):
    """
    Converts a list of triplets into a directed graph.

    Args:
        triplets (TripletExtraction): An object containing extracted triplets.

    Returns:
        networkx.DiGraph: A directed graph representing the triplets.
    """
    G = nx.DiGraph()
    for triplet in triplets.triplets:
        G.add_node(triplet.source.strip(), label=triplet.source.strip())
        G.add_node(triplet.target.strip(), label=triplet.target.strip())
        G.add_edge(triplet.source.strip(), triplet.target.strip(), label=triplet.edge.strip())
    return G
