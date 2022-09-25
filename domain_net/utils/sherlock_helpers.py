import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sherlock.features.preprocessing import (
    extract_features
)
from .graph_helpers import get_attribute_of_instance

def get_column_node_to_semantic_type_dict(data_dir, model, input_data_file_type='csv'):
    '''
    Given a directory `data_dir` containing a list of csv (i.e., tables) files and the pre-trained Sherlock `model`
    predict the semantic type for each column node (i.e, for each column in each specified table)
    '''
    # Dictionary mapping each column node to the predicted label using Sherlock
    column_node_to_semantic_type_dict = {}

    for filename in tqdm(os.listdir(data_dir)):
        if input_data_file_type=='csv':
            df = pd.read_csv(data_dir+filename, keep_default_na=False, dtype=str)
        elif input_data_file_type=='tsv':
            df = pd.read_csv(data_dir+filename, keep_default_na=False, dtype=str, sep='\t')
        else:
            raise ValueError('input_data_file_type must be one of: csv or tsv')

        column_names=df.columns
        
        # Convert dataframe into a pandas series to be used as input for Sherlock
        rows = [df[column_name].tolist() for column_name in column_names]
        data_series = pd.Series(rows, name='values')
        
        # Extract features for the current table and save them at temporary.csv
        extract_features("../temporary.csv", data_series)
        feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)
        
        # Use Sherlock to predict the semantic types for each column
        predicted_labels = model.predict(feature_vectors, "sherlock")
        for column_name, label in zip(column_names, predicted_labels):
            column_node_to_semantic_type_dict[column_name+'_'+filename] = label

    return column_node_to_semantic_type_dict

def get_cell_node_to_semantic_type(graph, column_node_to_semantic_type_dict):
    '''
    Return a dictionary mapping each cell value in a graph to the list of its semantic types found using
    Sherlock.
    '''
    # Extract all cell nodes from the graph
    cell_nodes = {n for n, d in graph.nodes(data=True) if d['type']=='cell'}
    cell_node_to_semantic_type_dict = {}

    # For each cell node find all the column nodes it is connected to and extract their semantic types
    for node in cell_nodes:
        semantic_types = set()
        column_nodes = get_attribute_of_instance(graph, instance_node=node)
        for column_node in column_nodes:
            semantic_types.add(column_node_to_semantic_type_dict[column_node])
        
        cell_node_to_semantic_type_dict[node]=semantic_types
 
    return cell_node_to_semantic_type_dict

def get_predicted_homographs(cell_node_to_semantic_type_dict):
    '''
    Returns a set of the cell node that were predicted to be homographs 
    (i.e., they were assigned to more than 1 semantic type)
    '''
    predicted_homographs = set()
    for cell_node in cell_node_to_semantic_type_dict:
        if len(cell_node_to_semantic_type_dict[cell_node]) > 1:
            predicted_homographs.add(cell_node)
    return predicted_homographs