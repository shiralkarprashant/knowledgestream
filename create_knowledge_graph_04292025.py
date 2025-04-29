"""
Creates files representing a knowledge graph (KG) or multi-graph,
using abbreviated triples preprocessed earlier.

Inputs: 
a. abbrev_yago_filtered.nt

Outputs:
a. nodes.txt: a set of nodes
b. relations.txt: a set of relations. 
c. adjacency.npy: a binary file of (S, O, P) indices per line.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from time import time
from os.path import basename, abspath, dirname, join, exists, expanduser
from pandas import DataFrame
from argparse import ArgumentParser


def parse(nt_file, outpath):
    """
    Scans the N-triples file representing an input RDF graph, 
    and saves the following to disk
    
    a. nodes.txt: a set of nodes
    b. relations.txt: a set of relations. 
    c. triples_abbrev.nt: abbreviated triples
    
    Parameters:
    -----------
    nt_file: str
        Absolute path to the N-triples file.
    outpath: str
        Directory name where nodes, relations and abbreviated triples
        should be stored.

    Returns:
    --------
    nodes_path, relations_path: tuple of str
        Path to the files containing nodes and relations.
    """
    nodes_path = join(outpath, 'nodes.txt')
    relations_path = join(outpath, 'relations.txt')

    print(f'Reading triples file: {basename(nt_file)}')
    df = pd.read_table(nt_file, sep=" ", header=None)
    ntriples = df.shape[0]
    print(f'Success! #triples: {ntriples}')

    nodes = set(df.iloc[:, 0]) | set(df.iloc[:, 2])  # sub & obj
    relations = set(df.iloc[:, 1])
    print(f"#Nodes (unique): {len(nodes)}")
    print(f"#Relations (unique): {len(relations)}")

    # save nodes
    nodes = sorted(list(nodes))
    nodes_df = DataFrame.from_dict({'node_id': np.arange(len(nodes)), 'node_name': nodes})
    nodes_df = nodes_df[['node_id', 'node_name']]
    nodes_df.to_csv(nodes_path, sep=" ", header=False, index=False)
    print(f"Saved nodes to {nodes_path}.")

    # save relations
    relations = sorted(list(relations))
    relations_df = DataFrame.from_dict({'rid': np.arange(len(relations)), 'r_name': relations})
    relations_df = relations_df[['rid', 'r_name']]
    relations_df.to_csv(relations_path, sep=" ", header=False, index=False)
    print(f"Saved relations to {relations_path}.")
    return nodes_path, relations_path


def create_edges(triples_abbrev_path, nodes_path, relations_path, outpath):
    """
    Creates coordinates for edges (sub, obj, relation) for triples in the triples_abbrev_path,
    using unique node list in nodes_path and unique relations in relations_path.
    The coordinates are saved to disk as 'adjacency.npy'.
    
    ** Note: Coordinates are in (S, O, P), instead of the usual (S, P, O) format.

    Parameters:
    -----------
    triples_abbrev_path: str
        Path to the abbreviated triples file.
    nodes_path: str
        Path to the unique node list.
    relations_path: str
        Path to the unique relations list.
    outpath: str
        Directory where adjacency (coordinates) file needs to be stored.
        The file will be named 'adjacency.npy'
    
    Returns:
    --------
    adjacency_file: str
        Path of the adjacency file (.npy).
    """
    nodes = pd.read_table(nodes_path, sep=" ", header=None)
    nodes.columns = ['node_id', 'node_name']
    print(f'Nodes read success! #Nodes: {nodes.shape[0]}')

    relations = pd.read_table(relations_path, sep=" ", header=None)
    relations.columns = ['relation_id', 'relation']
    print(f'Relations read success! #Relations: {relations.shape[0]}')

    triples = pd.read_table(triples_abbrev_path, sep=" ", header=None)
    triples = triples.iloc[:, :3]  # exclude '.'
    triples.columns = ['sub', 'pred', 'obj']
    ntriples = triples.shape[0]
    print(f'Triples read success! #Triples: {ntriples}')

    tmp1 = pd.merge(triples, nodes, how="inner",
                    left_on='sub', right_on='node_name')
    tmp1 = tmp1.rename(columns={'node_id': 'sid'})
    
    tmp2 = pd.merge(tmp1, nodes, how="inner",
                    left_on='obj', right_on='node_name')
    tmp2 = tmp2.rename(columns={'node_id': 'oid'})

    tmp3 = pd.merge(tmp2, relations, how="inner",
                    left_on='pred', right_on='relation')
    tmp3 = tmp3.rename(columns={'relation_id': 'pid'})
    triples_coo = tmp3[['sid', 'oid', 'pid']].values  # S-O-P format
    triples_coo = triples_coo[np.argsort(triples_coo[:, 2]), ]

    adjacency_file = os.path.join(outpath, 'adjacency.npy')
    np.save(adjacency_file, triples_coo)
    print(f"Edges saved! at {adjacency_file}.")
    return adjacency_file


def create_multigraph(nt_file, destination):
    """
    Creates a multi-relational graph from the N-triples file.
    
    Following files are generated:
    a. nodes.txt: a set of nodes
    b. relations.txt: a set of relations. 
    c. triples_abbrev.nt: abbreviated triples
    d. adjacency.npy: a binary file of (S, O, P) indices per line.

    Parameters:
    -----------
    nt_file: str
        Absolute path to the N-triples file.
    destination: str
        Absolute path to the directory where output files 
        will be stored.
    """
    # 1. Parse RDF graphs (N-triples/.nt format)
    t1 = time()
    print('1. Extracting nodes, relations and abbreviated triples..')
    nodes_path, relations_path = parse(nt_file, destination)
    print(f"Time taken: {time() - t1:.4f} secs.\n")
    sys.stdout.flush()

    # 2. Create coordinate relational edges (i, j, k)
    t1 = time()
    print("2. Creating coordinates for edges..")
    adjacency_path = create_edges(nt_file, nodes_path, relations_path, destination)
    print(f"Time taken to create edge coordinates: {time() - t1:.4f} secs.")


if __name__ == '__main__':
    """
    Example call:

    python create_multigraph.py -D ~/Projects/truthy_data/yago/yago3/processed/kg/
        ~/Projects/truthy_data/yago/yago3/processed/kg/abbrev_yago_filtered.nt  
        > create_multigraph.log
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('nt_file', metavar='ntriples', help='N-Triples file')
    parser.add_argument('-D', '--destination',
                       help='Absolute destination path. Otherwise, same as ntriples file.')
    args = parser.parse_args()
    print()

    # Parameters
    nt_file = abspath(expanduser(args.nt_file))
    if not exists(nt_file):
        raise Exception(f'N-triples file not found: {nt_file}')
    if args.destination is None:
        destination = abspath(dirname(nt_file))
    else:
        destination = abspath(expanduser(args.destination))
    print('## INPUT:')
    print(f'N-triples file: {nt_file}')
    print(f'Output path:  {destination}')
    print()

    create_multigraph(nt_file, destination)

    print("\nDone!\n\n")