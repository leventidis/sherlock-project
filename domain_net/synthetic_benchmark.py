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

def main():
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
    data_dir="data/synthetic_benchmark/"
    graph = pickle.load(open('graph_representations/synthetic_benchmark/bipartite.graph', 'rb'))

    # Find the semantic types for each column
    column_node_to_semantic_type_dict = utils.sherlock_helpers.get_column_node_to_semantic_type_dict(data_dir=data_dir, model=model)
    with open('output/synthetic_benchmark/column_node_to_semantic_type_dict.pickle', 'wb') as handle:
        pickle.dump(column_node_to_semantic_type_dict, handle)

    # Find the semantic types for each cell node
    cell_node_to_semantic_type_dict = utils.sherlock_helpers.get_cell_node_to_semantic_type(graph, column_node_to_semantic_type_dict)
    predicted_homographs = utils.sherlock_helpers.get_predicted_homographs(cell_node_to_semantic_type_dict)

    with open('output/synthetic_benchmark/cell_node_to_semantic_type_dict.pickle', 'wb') as handle:
        pickle.dump(cell_node_to_semantic_type_dict, handle)
    with open('output/synthetic_benchmark/predicted_homographs.pickle', 'wb') as handle:
        pickle.dump(predicted_homographs, handle)

if __name__ == "__main__":
    main()