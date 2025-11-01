import os
import json
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

import torch
import pickle
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset


class AGQADataset(Dataset):
    # tested only for the AGQA2.0
    def __init__(self, split, lm_model="mbert", seq_limit=float('inf')):
        super().__init__()
        self.root_path = os.environ["AGQA_ROOT"]

        self.split = split
        assert split in ["train", "test"]

        logger.info(f"Loading QA grounding")
        qa2sg_data_path = \
            f"{self.root_path}/preprocessed_{lm_model}/{self.split}/qa2sg.pkl"
        with open(qa2sg_data_path, "rb") as file:
            self.qa2sg = list(pickle.load(file).values())
        logger.info(f"G len {len(self.qa2sg)}")

        logger.info(f"Loading QA")
        if seq_limit == float('inf'):
            self.load_idx = list(range(len(self.qa2sg)))
        else:
            self.load_idx = [idx for idx, item in enumerate(self.qa2sg) if 0 < len(item) <= seq_limit]
        logger.warning(f"For thresh={seq_limit}: num seqs={len(self.load_idx)}/{len(self.qa2sg)}")
        qa_data_path = f"{self.root_path}/data/AGQA_balanced/{self.split}_balanced.txt"
        self.qa_data = list(json.load(open(qa_data_path, mode='r', encoding='utf8')).values())

        logger.info(f"Loading graphs")
        self.graphs = {file.split(".")[0]: 
            torch.load(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/graphs/{file}", weights_only=False) for 
                file in os.listdir(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/graphs/")}
        logger.info(f"G len {len(self.graphs)}")

        logger.info(f"Loading decsriptions")
        self.descs = {}
        for file in tqdm(os.listdir(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/descs/")):  
            with open(f"{self.root_path}/preprocessed_{lm_model}/{self.split}/descs/{file}", "rb") as f:
                self.descs[file.split(".")[0]] = pickle.load(f)

    def __len__(self):
        return len(self.load_idx)

    def __getitem__(self, index):
        item = self.qa_data[self.load_idx[index]]
        graphs = [g for r, g in self.graphs[item["video_id"]].items() if r in self.qa2sg[self.load_idx[index]]]
        descs = [d for r, d in self.descs[item["video_id"]].items() if r in self.qa2sg[self.load_idx[index]]]

        orig_idxs = [int(r[0]) for r, g in self.graphs[item["video_id"]].items() if r in self.qa2sg[self.load_idx[index]]]
        assert len(orig_idxs) == len(graphs)
        assert len(graphs) != 0

        if len(orig_idxs) == 1: # ZeroDivision Error
           orig_idxs = [0]
        else:
            norm_min, norm_max = min(orig_idxs), max(orig_idxs)
            orig_idxs = [(i - norm_min) / (norm_max - norm_min) for i in orig_idxs]
        descs = [f"Start of graph {round(i, 1)}\n{d}\nEnd of graph {round(i, 1)}" for i, d in zip(orig_idxs, descs)]
        
        return {
            "question": item["question"],
            "question_type": {"bo_type": item["ans_type"],
                              "reasoning": item["global"],
                              "semantic": item["semantic"],
                              "structural": item["structural"]},
            "answer": item["answer"].lower(),
            "graphs": graphs,
            "orig_idxs": orig_idxs,
            "decs": "\n".join(descs)
        }
