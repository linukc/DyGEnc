#!/bin/bash
python3 train.py --dataset_name=star --exp_name=table_1_3b_tpe tpe=tpe
python3 predict.py --dataset_name=star --exp_name=table_1_3b_tpe tpe=tpe
python3 eval.py --dataset_name=star --exp_name=table_1_3b_tpe

python3 train.py --dataset_name=star --exp_name=table_1_3b_ape tpe=ape
python3 predict.py --dataset_name=star --exp_name=table_1_3b_ape tpe=ape
python3 eval.py --dataset_name=star --exp_name=table_1_3b_ape

python3 train.py --dataset_name=star --exp_name=table_1_3b_rpe tpe=rpe
python3 predict.py --dataset_name=star --exp_name=table_1_3b_rpe tpe=rpe
python3 eval.py --dataset_name=star --exp_name=table_1_3b_rpe

python3 train.py --dataset_name=star --exp_name=table_1_8b_tpe tpe=tpe llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_1_8b_tpe tpe=tpe llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_1_8b_tpe

python3 train.py --dataset_name=star --exp_name=table_1_8b_ape tpe=ape llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_1_8b_ape tpe=ape llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_1_8b_ape

python3 train.py --dataset_name=star --exp_name=table_1_8b_rpe tpe=rpe llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table_1_8b_rpe tpe=rpe llm_model_path="meta-llama/Llama-3.1-8b" val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table_1_8b_rpe
