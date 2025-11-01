import os
import json
import string
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

import torch
import pickle
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset


def load_graph(idx, root_path, lm_model, split):
    file_path = f"{root_path}/preprocessed_{lm_model}/{split}/graphs/{idx}.pt"
    return torch.load(file_path)

class StarDataset(Dataset):
    def __init__(self, split, lm_model="mbert", seq_limit=float('inf')):
        super().__init__()
        self.root_path = os.environ["STAR_ROOT"]

        self.split = split
        assert split in ["train", "val"]

        split_metainfo_path = \
            f"{self.root_path}/preprocessed_{lm_model}/{self.split}/num_graphs_metadata.pkl"
        with open(split_metainfo_path, "rb") as file:
            self.split_metainfo = pickle.load(file) # each value in list describe num of graphs in seq

        self.load_idx = [idx for idx, num in enumerate(self.split_metainfo) if 0 < num <= seq_limit]
        logger.warning(f"For thresh={seq_limit}: num seqs={len(self.load_idx)}/{len(self.split_metainfo)}")

        idx_metainfo_path = \
            f"{self.root_path}/preprocessed_{lm_model}/{self.split}/original_indices.pkl"
        with open(idx_metainfo_path, 'rb') as file:
            self.idx_metainfo = pickle.load(file) # original indices of G in seq for temporal encoding
        self.idx_metainfo = [self.idx_metainfo[i] for i in self.load_idx]
        
        self.data_path = f"{self.root_path}/data/STAR_{self.split}.json"
        with open(self.data_path, "rt") as file:
            self.json_data = json.load(file)
        self.q = [self.json_data[i]["question"] for i in self.load_idx]
        logger.info(f"Q len {len(self.q)}")
        self.a = [self.json_data[i]["answer"].translate(
            str.maketrans('', '', string.punctuation.replace("/", ""))).lower().replace("the ", "").strip() 
                for i in self.load_idx]
        self.a_variants = [[j["choice"].replace("The", "").replace(".", "").strip() for j in self.json_data[i]["choices"]] for i in self.load_idx]
        self.q_t = [self.json_data[i]["question_id"].split("_")[0] for i in self.load_idx]
        logger.info(f"A len {len(self.a)}")
    
        logger.info(f"Loading graphs")
        self.graphs = [torch.load(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/graphs/{idx}.pt")
            for idx in tqdm(self.load_idx)]
        logger.info(f"G len {len(self.graphs)}")

        logger.info(f"Loading decsriptions")
        self.descs = []
        for idx in tqdm(self.load_idx):    
            with open(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/descs/{idx}.pkl", "rb") as f:
                self.descs.append(pickle.load(f))
    
    def __len__(self):
        return len(self.load_idx)

    def __getitem__(self, index):
        return {
            "question": self.q[index],
            "question_type": self.q_t[index],
            "answer": self.a[index],
            "a_variants": self.a_variants[index],
            "graphs": self.graphs[index],
            "orig_idxs": self.idx_metainfo[index],
            "decs": self.descs[index]
        }
        