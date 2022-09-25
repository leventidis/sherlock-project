#!/bin/bash

# Shell script that identifies the homographs in the TUS benchmark

input_dir=data/D4-Education/
output_dir=output/D4-Education/
graph=graph_representations/D4-Education/bipartite.graph

python sherlock_predicted_homographs.py \
--input_dir $input_dir --output_dir $output_dir --graph $graph --input_data_file_type tsv