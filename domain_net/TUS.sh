#!/bin/bash

# Shell script that identifies the homographs in the TUS benchmark

input_dir=data/TUS/csvfiles/
output_dir=output/TUS/
graph=graph_representations/TUS/bipartite.graph

python sherlock_predicted_homographs.py \
--input_dir $input_dir --output_dir $output_dir --graph $graph