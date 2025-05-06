import os
import json
import pickle

import torch
from loguru import logger
from torch.utils.data import Dataset

from src.datasets.utils import retrieval_via_pcst


class UserDataset(Dataset):
    def __init__(self, lm_model="mbert", do_retrieve=False, **kwargs):
        super().__init__()
        self.root_path = os.environ["CUSTOM_ROOT"]
        self.lm_model = lm_model
        self.do_retrieve = do_retrieve
        logger.info(f"Do retrieve is set={self.do_retrieve}.")

        logger.info("Loading QA")
        with open(f"{self.root_path}/qa.json") as file:
            self.qa = json.load(file)

        logger.info("Loading q_embed")
        self.q_embed = torch.load(f"{self.root_path}/preprocessed_{lm_model}/q_embed.pt")

        logger.info("Loading qraph idxs")
        with open(f"{self.root_path}/preprocessed_{lm_model}/original_indices.pkl", 'rb') as f:
            self.idx_metainfo = pickle.load(f)

        logger.info("Loading graphs")
        self.graphs = torch.load(f"{self.root_path}/preprocessed_{lm_model}/graphs/pyg.pt")
        self.nx_graphs = []
        for graph_name in sorted(os.listdir(f"{self.root_path}/sg_pickle/"), key=lambda i: int(i.split("_")[1])):
            with open(f"{self.root_path}/sg_pickle/{graph_name}", 'rb') as f:
                self.nx_graphs.append(pickle.load(f))


    def __getitem__(self, index):
        all_graphs = None
        if self.do_retrieve:
            os.makedirs(f"{self.root_path}/preprocessed_{self.lm_model}/retrieved_vis_q{index}", exist_ok=True)
            all_graphs = []
            for i, (graph, graph_nx) in enumerate(zip(self.graphs, self.nx_graphs)):
                new_pyg_graph = retrieval_via_pcst(graph, self.q_embed[index])
                all_graphs.append(new_pyg_graph)
        else:
            all_graphs = self.graphs

        return {
            "question": self.qa[index]["question"],
            "question_type": "custom",
            "answer": self.qa[index]["answer"],
            "graphs": all_graphs,
            "orig_idxs": self.idx_metainfo[0]
        }

    def __len__(self):
        return len(self.qa)
