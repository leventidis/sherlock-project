# Use Sherlock to identify all the homographs from the TUS-Injected benchmark

import pickle5 as pickle
import utils
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import networkx as nx

from tqdm import tqdm
from pathlib import Path

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
    

    # Loop over the various cardinalities used by the injected datasets
    base_dir="data/TUS_injected_homographs/"
    for dir in sorted(os.listdir(base_dir)):
                
        # Load the TUS dataset and identify homographs
        data_dir="data/TUS_injected_homographs/"+dir+"/"
        graph = pickle.load(open('graph_representations/TUS_injected_homographs/'+dir+'/bipartite.graph', 'rb'))
        Path("output/TUS_injected_homographs/"+dir+'/').mkdir(parents=True, exist_ok=True)

        # Find the semantic types for each column
        column_node_to_semantic_type_dict = utils.sherlock_helpers.get_column_node_to_semantic_type_dict(data_dir=data_dir, model=model)
        with open('output/TUS_injected_homographs/'+dir+'/column_node_to_semantic_type_dict.pickle', 'wb') as handle:
            pickle.dump(column_node_to_semantic_type_dict, handle)

        # Find the semantic types for each cell node
        cell_node_to_semantic_type_dict = utils.sherlock_helpers.get_cell_node_to_semantic_type(graph, column_node_to_semantic_type_dict)
        predicted_homographs = utils.sherlock_helpers.get_predicted_homographs(cell_node_to_semantic_type_dict)

        with open('output/TUS_injected_homographs/'+dir+'/cell_node_to_semantic_type_dict.pickle', 'wb') as handle:
            pickle.dump(cell_node_to_semantic_type_dict, handle)
        with open('output/TUS_injected_homographs/' + dir + '/predicted_homographs.pickle', 'wb') as handle:
            pickle.dump(predicted_homographs, handle)


if __name__ == "__main__":
    main()