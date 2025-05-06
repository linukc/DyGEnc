#!/bin/bash
python3 train.py --dataset_name=star --exp_name=table3_3b_wo_gnn_tpe_qformer enable_gnn=False enable_tpe=False enable_qformer=False train_batch_size=1 lr=0.000005 val_batch_size=1 num_epochs=3
python3 predict.py --dataset_name=star --exp_name=table3_3b_wo_gnn_tpe_qformer enable_gnn=False enable_tpe=False enable_qformer=False val_batch_size=4
python3 eval.py --dataset_name=star --exp_name=table3_3b_wo_gnn_tpe_qformer
python3 train.py --dataset_name=star --exp_name=table3_8b_wo_gnn_tpe_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=1 lr=0.000005 enable_gnn=False enable_tpe=False enable_qformer=False val_batch_size=1 num_epochs=3
python3 predict.py --dataset_name=star --exp_name=table3_8b_wo_gnn_tpe_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=1 enable_gnn=False enable_tpe=False enable_qformer=False val_batch_size=4
python3 eval.py --dataset_name=star --exp_name=table3_8b_wo_gnn_tpe_qformer

python3 train.py --dataset_name=star --exp_name=table3_3b_wo_tpe_qformer enable_tpe=False enable_qformer=False train_batch_size=8 lr=0.00001 val_batch_size=128
python3 predict.py --dataset_name=star --exp_name=table3_3b_wo_tpe_qformer enable_tpe=False enable_qformer=False val_batch_size=128
python3 eval.py --dataset_name=star --exp_name=table3_3b_wo_tpe_qformer
python3 train.py --dataset_name=star --exp_name=table3_8b_wo_tpe_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=8 lr=0.00001 enable_tpe=False enable_qformer=False val_batch_size=128
python3 predict.py --dataset_name=star --exp_name=table3_8b_wo_tpe_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=8 enable_tpe=False enable_qformer=False val_batch_size=128
python3 eval.py --dataset_name=star --exp_name=table3_8b_wo_tpe_qformer

python3 train.py --dataset_name=star --exp_name=table3_3b_wo_tpe enable_tpe=False
python3 predict.py --dataset_name=star --exp_name=table3_3b_wo_tpe enable_tpe=False
python3 eval.py --dataset_name=star --exp_name=table3_3b_wo_tpe
python3 train.py --dataset_name=star --exp_name=table3_8b_wo_tpe llm_model_path="meta-llama/Llama-3.1-8b" enable_tpe=False val_batch_size=384
python3 predict.py --dataset_name=star --exp_name=table3_8b_wo_tpe llm_model_path="meta-llama/Llama-3.1-8b" enable_tpe=False val_batch_size=384
python3 eval.py --dataset_name=star --exp_name=table3_8b_wo_tpe

python3 train.py --dataset_name=star --exp_name=table3_3b_wo_qformerenable_qformer=False train_batch_size=8 lr=0.00001 val_batch_size=128
python3 predict.py --dataset_name=star --exp_name=table3_3b_wo_qformer enable_qformer=False train_batch_size=8 val_batch_size=128
python3 eval.py --dataset_name=star --exp_name=table3_3b_wo_qformer
python3 train.py --dataset_name=star --exp_name=table3_8b_wo_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=8 lr=0.00001 enable_qformer=False val_batch_size=128
python3 predict.py --dataset_name=star --exp_name=table3_8b_wo_qformer llm_model_path="meta-llama/Llama-3.1-8b" train_batch_size=8 enable_qformer=False val_batch_size=128
python3 eval.py --dataset_name=star --exp_name=table3_8b_wo_qformer
