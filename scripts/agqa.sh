#!/bin/bash
python3 train.py --dataset_name=agqa --exp_name=table5_3b num_epochs=2
python3 train.py --dataset_name=agqa --exp_name=table5_8b llm_model_path="meta-llama/Llama-3.1-8b" num_epochs=2 val_batch_size=384