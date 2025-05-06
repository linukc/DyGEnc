#!/bin/bash
python3 train.py --dataset_name=star --exp_name=table_2_3b_numq1 q_num_latent=1
python3 predict.py --dataset_name=star --exp_name=table_2_3b_numq1 q_num_latent=1
python3 eval.py --dataset_name=star --exp_name=table_2_3b_numq1

python3 train.py --dataset_name=star --exp_name=table_2_3b_numq2 q_num_latent=2
python3 predict.py --dataset_name=star --exp_name=table_2_3b_numq2 q_num_latent=2
python3 eval.py --dataset_name=star --exp_name=table_2_3b_numq2

python3 train.py --dataset_name=star --exp_name=table_2_3b_numq4 q_num_latent=4
python3 predict.py --dataset_name=star --exp_name=table_2_3b_numq4 q_num_latent=4
python3 eval.py --dataset_name=star --exp_name=table_2_3b_numq4

python3 train.py --dataset_name=star --exp_name=table_2_3b_numq16 q_num_latent=16
python3 predict.py --dataset_name=star --exp_name=table_2_3b_numq16 q_num_latent=16
python3 eval.py --dataset_name=star --exp_name=table_2_3b_numq16

python3 train.py --dataset_name=star --exp_name=table_2_8b_numq1 q_num_latent=1 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_2_8b_numq1 q_num_latent=1 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_2_8b_numq1

python3 train.py --dataset_name=star --exp_name=table_2_8b_numq2 q_num_latent=2 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_2_8b_numq2 q_num_latent=2 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_2_8b_numq2

python3 train.py --dataset_name=star --exp_name=table_2_8b_numq4 q_num_latent=4 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_2_8b_numq4 q_num_latent=4 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_2_8b_numq4

python3 train.py --dataset_name=star --exp_name=table_2_8b_numq16 q_num_latent=16 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_2_8b_numq16 q_num_latent=16 llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_2_8b_numq16
