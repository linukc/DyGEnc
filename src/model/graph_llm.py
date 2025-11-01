import contextlib

import torch
import torch.nn as nn
from loguru import logger
from torch_scatter import scatter
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.model.qformer import QFormer
from src.model.gnn import load_gnn_model
from src.model.encoding import load_encoding


# llama3 specific
BOS = '<|begin_of_text|>'
BOS_USER = '<|start_header_id|>user<|end_header_id|>'
EOS_USER = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
EOS = '<|end_of_text|>'
IGNORE_INDEX = -100


class DGMap3d(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.prompt = "Based on scene graph, "
        self.tail_prompt = ", with latent graph features "

        logger.info('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '80GiB'},
            "device_map": "cuda",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        self.special_tokens_dict = {
            "additional_special_tokens":
                ["<graph>", "</graph>"]
        }
        self.tokenizer.add_special_tokens(self.special_tokens_dict)
        model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Setup LLAMA with LORA!")
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        self.word_embedding = model.get_input_embeddings()
        self.model = model
        logger.info('Finish loading LLAMA!')

        if self.cfg.enable_gnn:
            self.graph_encoder = load_gnn_model[cfg.gnn_model_name](
                in_channels=cfg.gnn_in_channels, # should be divisible by num_heads
                hidden_channels=cfg.gnn_hidden_channels, # should be divisible by num_heads
                out_channels=cfg.gnn_out_channels,
                num_layers=cfg.gnn_num_layers,
                num_heads=cfg.gnn_num_heads,
                dropout=cfg.gnn_dropout,
            ).to(self.model.device)

            self.projector = nn.Sequential(
                nn.Linear(cfg.proj_in_channels, cfg.proj_hid_channels),
                nn.GELU(),
                nn.Linear(cfg.proj_hid_channels, self.word_embedding.weight.shape[1]),
            ).to(self.model.device)

        assert type(self.cfg.enable_qformer) == bool
        if self.cfg.enable_qformer:
            self.qformer = QFormer(
                num_query_tokens=cfg.q_num_latent,    # e.g., num of learned queries
                hidden_dim=cfg.q_hid_dim,        # embedding dimension
                num_layers=cfg.q_num_layers,          # 2 transformer layers
                num_heads=cfg.q_num_heads,           # multi-head attention with 4 heads
                ff_dim=cfg.q_ff_dim             # feed-forward hidden size
            ).to(self.model.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graphs']
        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)

        # mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        # Split back into the original structure
        outputs_per_seq = []
        idx = 0
        for length in samples['seq_lengths']:
            # For each element, slice out the right number of embeddings
            seq_embs = g_embeds[idx : idx + length]
            idx += length
            outputs_per_seq.append(seq_embs)

        return outputs_per_seq

    def collate_embeddings_with_temporal_enc(self, list_of_tensors, positions):
        """
        list_of_tensors: List[Tensor], where each tensor has shape (L_i, hidden_dim).
        positions: List[List[int]], where each value is a idx in the original seq from 0 to N

        Returns:
            padded_tensor: (B, L_max, hidden_dim)
            mask:          (B, L_max)  with True = valid, False = padded
        """
        # 1) Find max length L_max
        max_len = max(t.size(0) for t in list_of_tensors)
        batch_size = len(list_of_tensors)
        hidden_dim = list_of_tensors[0].size(1)

        assert type(self.cfg.enable_tpe) == bool
        if self.cfg.enable_tpe:
            temporal_encoder = load_encoding[self.cfg.tpe](out_channels=hidden_dim).to(self.model.device)

        # 2) Allocate
        padded_tensor = torch.zeros(batch_size, max_len, hidden_dim, device=self.model.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.model.device)

        # 3) Copy data & create mask
        for i, (emb, pos) in enumerate(zip(list_of_tensors, positions)):
            length = emb.size(0)

            if self.cfg.enable_tpe:
                timestamps = torch.tensor(pos, dtype=torch.float, device=self.model.device)
                assert timestamps.max().item() <= 1.0
                assert timestamps.numel() != 0
    
                if self.cfg.tpe != "rpe":
                    time_emb = temporal_encoder(timestamps)
                    padded_tensor[i, :length] += time_emb
                if self.cfg.tpe == "rpe":
                    padded_tensor[i, :length] += temporal_encoder(emb, timestamps)
                else:
                    padded_tensor[i, :length] += emb
            else:
                padded_tensor[i, :length] += emb
            mask[i, :length] = True  # valid positions

        return padded_tensor, mask

    def prepare_train_input(self, samples):
        # encode description, questions and labels
        prompt_embeds = self.word_embedding(self.tokenizer(self.prompt, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        tail_prompt_embeds = self.word_embedding(self.tokenizer(self.tail_prompt, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        labels = self.tokenizer(samples["answer"], add_special_tokens=False)

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        bos_user_embeds = self.word_embedding(self.tokenizer(BOS_USER, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        sg_start_embeds = self.word_embedding(
            self.tokenizer("<graph>", add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        sg_end_embeds = self.word_embedding(
            self.tokenizer("</graph>", add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))

        # encode graphs
        if self.cfg.enable_gnn:
            graph_embeds = self.encode_graphs(samples)
            graph_embeds, pad_mask = self.collate_embeddings_with_temporal_enc(graph_embeds, positions=samples["orig_idxs"])
            assert type(self.cfg.enable_qformer) == bool
            if self.cfg.enable_qformer:
                graph_embeds = self.qformer(graph_embeds, pad_mask)
            graph_embeds = self.projector(graph_embeds)
        #else:
        decsc = self.tokenizer(samples["decs"], add_special_tokens=False)
        graph_embeds_decsc = decsc.input_ids#.to(self.model.device)

        batch_size = len(samples["answer"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i] + eos_tokens.input_ids
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if not self.cfg.enable_qformer and self.cfg.enable_gnn: # apply unpad
                gemb = graph_embeds[i][pad_mask[i]]
            elif not self.cfg.enable_gnn:
                gemb = self.word_embedding(torch.tensor(graph_embeds[i]).to(self.model.device))
            else:
                gemb = graph_embeds[i]

            gemb_decs = self.word_embedding(torch.tensor(graph_embeds_decsc[i]).to(self.model.device))

            inputs_embeds = torch.cat([bos_embeds, bos_user_embeds, prompt_embeds, gemb_decs, 
                                       tail_prompt_embeds, sg_start_embeds, gemb, sg_end_embeds, 
                                       inputs_embeds], dim=0) 

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        return inputs_embeds, attention_mask, label_input_ids

    def forward(self, inputs_embeds, attention_mask, label_input_ids):
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    def inference(self, samples):
        prompt_embeds = self.word_embedding(self.tokenizer(self.prompt, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        tail_prompt_embeds = self.word_embedding(self.tokenizer(self.tail_prompt, add_special_tokens=False, return_tensors='pt').input_ids[0].to(
            self.model.device))
        questions = self.tokenizer(samples["question"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        bos_user_embeds = self.word_embedding(self.tokenizer(BOS_USER, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        sg_start_embeds = self.word_embedding(self.tokenizer(
             "<graph>", add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        sg_end_embeds = self.word_embedding(self.tokenizer(
             "</graph>", add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))

        # encode graphs
        if self.cfg.enable_gnn:
             graph_embeds = self.encode_graphs(samples)
             graph_embeds, pad_mask = self.collate_embeddings_with_temporal_enc(graph_embeds, positions=samples["orig_idxs"])
             assert type(self.cfg.enable_qformer) == bool
             if self.cfg.enable_qformer:
                 graph_embeds = self.qformer(graph_embeds, pad_mask)
             graph_embeds = self.projector(graph_embeds)
        #else:
        decsc = self.tokenizer(samples["decs"], add_special_tokens=False)
        graph_embeds_decsc = decsc.input_ids #.to(self.model.device)

        batch_size = len(samples["answer"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if not self.cfg.enable_qformer and self.cfg.enable_gnn: # apply unpad
                 gemb = graph_embeds[i][pad_mask[i]]
            elif not self.cfg.enable_gnn:
                 gemb = self.word_embedding(torch.tensor(graph_embeds[i]).to(self.model.device))
            else:
                 gemb = graph_embeds[i]

            gemb_decs = self.word_embedding(torch.tensor(graph_embeds_decsc[i]).to(self.model.device))
            
            inputs_embeds = torch.cat([bos_embeds, bos_user_embeds, prompt_embeds, gemb_decs, 
                                       tail_prompt_embeds, sg_start_embeds, gemb, sg_end_embeds, 
                                       inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.cfg.max_new_tokens,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {
                'pred': pred,
                'answer': samples['answer'],
                'question': samples['question'],
                'question_type': samples['question_type']
               }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
