#!/bin/bash
python3 train.py --dataset_name=star --exp_name=table4_3b
python3 train.py --dataset_name=star --exp_name=table4_8b llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384