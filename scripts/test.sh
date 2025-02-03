#!/bin/bash

source ~/.bashrc
conda activate /home/tianyizhou/anaconda3/envs/math/myenv

# Dataset and model details for test run
dataset="Onlydrinkwater/1000addition"
model="meta-llama/Llama-3.2-1B-Instruct"
batch_size=32         # Reduced batch size for testing
max_train_samples=10000  # Use a smaller subset of the dataset
lr=5e-4               # Single learning rate for test run
method="fne"          # Testing with the "fne" method

echo "Running test run for model $model on dataset $dataset with method $method, learning_rate=$lr"
cd ..
python main.py --lr $lr \
               --name test_run \
               --model_size_level 4 \
               --epochs 5 \
               --clip \
               --batch_size $batch_size \
               --model $model \
               --int_digit_len 10 \
               --frac_digit_len 0 \
               --dataset $dataset \
               --train_from_scratch \
               --method $method \
               --num_train_samples $max_train_samples
