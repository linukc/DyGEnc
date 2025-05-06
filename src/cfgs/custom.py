from dataclasses import dataclass


@dataclass
class Config:
    # general
    seed: int=18

    ### model
    llm_model_path: str="meta-llama/Llama-3.1-8b"
    gnn_model_name: str="gt"
    max_new_tokens: int=20
    #### lora
    lora_r: int=8
    lora_alpha: int=16
    lora_dropout: float=0.05
    #### graph_encoder
    enable_gnn: bool=True
    gnn_model_name: str="gt"
    gnn_in_channels: int=1028
    gnn_hidden_channels: int=1028
    gnn_out_channels: int=1024
    gnn_num_layers: int=4
    gnn_num_heads: int=4
    gnn_dropout: float=0.1
    #### temporal embeder
    enable_tpe: bool=True
    tpe: str="rpe"
    #### qformer
    enable_qformer: bool=True
    q_num_latent: int=1
    q_hid_dim: int=1024
    q_num_layers: int=2
    q_num_heads: int=4
    q_ff_dim: int=512
    #### projector
    proj_in_channels: int=1024
    proj_hid_channels: int=2048

    # val
    ### data
    val_split: str="test"
    val_lm_model: str="mbert"
    val_seq_limit: float=float("inf")

    ### val params
    val_batch_size: int=32
    val_num_workers: int=4
