#!/bin/bash

# Shell script that identifies the homographs in the synthetic dataset and the expanded synthetic dataset using Sherlock

#####----- Homograph prediction on the synthetic_benchmark -----#####
input_dir=data/synthetic_benchmark/
output_dir=output/synthetic_benchmark/
graph=graph_representations/synthetic_benchmark/bipartite.graph

python sherlock_predicted_homographs.py \
--input_dir $input_dir --output_dir $output_dir --graph $graph


#####----- Homograph prediction on the synthetic_benchmark_large -----#####
input_dir=data/synthetic_benchmark_large/
output_dir=output/synthetic_benchmark_large/
graph=graph_representations/synthetic_benchmark_large/bipartite.graph

python sherlock_predicted_homographs.py \
--input_dir $input_dir --output_dir $output_dir --graph $graph