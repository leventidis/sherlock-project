# Use Sherlock to identify all the homographs from the TUS benchmark

import pickle5 as pickle
import utils
import numpy as np
import pandas as pd
import pyarrow as pa
import networkx as nx

from tqdm import tqdm

from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings

import argparse
from pathlib import Path

def main(args):
    print("Loading embeddings...")
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()

    # Initialize the pre-trained sherlock model with its default parameters 
    print("Initializing the Sherlock Model...")
    model = SherlockModel();
    model.initialize_model_from_json(with_weights=True, model_id="sherlock");
    
    
    # Load the TUS dataset and identify homographs
    data_dir=args.input_dir
    graph = pickle.load(open(args.graph, 'rb'))

    # Find the semantic types for each column
    column_node_to_semantic_type_dict = utils.sherlock_helpers.get_column_node_to_semantic_type_dict(
        data_dir=data_dir, model=model, input_data_file_type=args.input_data_file_type
    )
    with open(args.output_dir + 'column_node_to_semantic_type_dict.pickle', 'wb') as handle:
        pickle.dump(column_node_to_semantic_type_dict, handle)

    # Find the semantic types for each cell node
    cell_node_to_semantic_type_dict = utils.sherlock_helpers.get_cell_node_to_semantic_type(graph, column_node_to_semantic_type_dict)
    predicted_homographs = utils.sherlock_helpers.get_predicted_homographs(cell_node_to_semantic_type_dict)

    with open(args.output_dir + 'cell_node_to_semantic_type_dict.pickle', 'wb') as handle:
        pickle.dump(cell_node_to_semantic_type_dict, handle)
    with open(args.output_dir + 'predicted_homographs.pickle', 'wb') as handle:
        pickle.dump(predicted_homographs, handle)



if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Given a set of tables use the Sherlock pre-trained model to predict the homographs')

    # Input directory containing the tables
    parser.add_argument('-i', '--input_dir', metavar='input_dir', required=True,
    help='Path to the input directory where input tables are present')

    # Output directory where output files are stored
    parser.add_argument('-o', '--output_dir', metavar='output_dir', required=True,
    help='Path to the output directory where output files and figures are stored.')

    # Input graph representation of the set of tables
    parser.add_argument('-g', '--graph', metavar='graph', required=True,
    help='Path to the Graph representation of the set of tables')

    # File format of the input raw data (i.e. the tables). One of {csv, tsv}
    parser.add_argument('-idft', '--input_data_file_type', choices=['csv', 'tsv'], default='csv',
    metavar='input_data_file_type', required=True,
    help='File format of the input raw data (i.e. the tables). One of {csv, tsv}')

    # Parse the arguments
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)