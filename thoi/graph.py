from typing import List
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd


def synergestic_graph(df: pd.DataFrame, marker_cols:List[int], weight_method: str = 'unweighted'):
    """
    @bierf: Compute the sinergestic graph to later compute other graph theory metrics
    
    @param df: dataframe with the synergestic nplets. The columns are i ... N and psize. where each row is a syneregestic nplet and the column i is True if the nplet contains the node i
    @param N: number of nodes in the graph

    @return: G: graph of connections where the weight of each edge is the synergestic connection of two nodes.

    if weight_method is 'unweighted' the weight of the edge is the number of synergestic nplets between the two nodes
    if weight_method is 'weighted' the weight of the edge is the number of synergestic nplets between the two nodes weighted by the psize column as 2*(1/psize)
    """

    N = len(marker_cols)

    # Create the graph of conections were the weight of the node is the number of synergestic nplets
    print('Creating the graph of connections...')
    G = nx.Graph()
    pbar_i = tqdm(range(N))
    for i in pbar_i:
        pbar_i.set_description(f'Node: {i}')
        G.add_node(i, variable=marker_cols[i])
        #pbar_j = tqdm(range(i+1, N+1), leave=False)
        for j in range(i+1, N):
            #pbar_j.set_description(f'Node: {j}')

            both_in_nplet = (df[marker_cols[i]] & df[marker_cols[j]]).astype(int).values

            # Count the number of nplets in common between the two nodes    
            if weight_method == 'unweighted':

                # count n rows where cols i and j are both True
                weight = np.sum(both_in_nplet)

            elif weight_method == 'weighted':
                
                # count n rows where cols i and j are both True weighted by the psize column as 2*(1/psize)
                order = (2/df['order']).values
                weight = np.sum(both_in_nplet * order)

            if weight > 0:
                G.add_edge(i, j, weight=weight)

    return G


def synergestic_centrality(G):

    # Compute the centrality of each node
    print('Computing the centrality of each node...')
    centrality = nx.eigenvector_centrality(G, weight='weight')

    # Add the centrality information to the graph nodes
    for i in G.nodes:
        G.nodes[i]['centrality'] = centrality[i]

    return G

def synergestic_clustering(G):
    clustering = nx.clustering(G, weight='weight')

    for i in G.nodes:
        G.nodes[i]['clustering'] = clustering[i]
    
    return G

def synergestic_degree(G):
    degree = nx.degree(G, weight='weight')

    for i in G.nodes:
        G.nodes[i]['degree'] = degree[i]
    
    return G