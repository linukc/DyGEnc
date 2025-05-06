import os
import json
import pickle

import torch
import networkx as nx
from tqdm import tqdm
from loguru import logger
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.utils.lm_modeling import load_model, load_text2embedding


root_path = os.environ["CUSTOM_ROOT"]
LPE_NUM = 4
logger.info(f"Setting lpe={LPE_NUM}")
### load embed model
MODEL_NAME = "mbert"
logger.info(f"Loading LM={MODEL_NAME}")
model, tokenizer, device = load_model[MODEL_NAME]()
logger.info(device)
text2embedding = load_text2embedding[MODEL_NAME]


def preprocess_graphs():
    logger.info("working with graphs")
    os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/", exist_ok=True)

    number_of_graphs = []
    original_indices = []

    os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/graphs", exist_ok=True)
    sg_seq_keys = sorted(os.listdir(f"{root_path}/sg_pickle/"), key=lambda i: int(i.split("_")[1]))
    sg_seq_nx = []
    # iterate over individual G in the sequence of graphs
    for sg in tqdm(sg_seq_keys):
        with open(f"{root_path}/sg_pickle/{sg}", "rb") as file:
            sg_seq_nx.append(pickle.load(file))

        # leave only unique graphs
        uniq_start_index = None
        for i in range(len(sg_seq_nx)):
            if len(list(nx.get_node_attributes(sg_seq_nx[i], "label").values())) > 0:
                uniq_start_index = i
                break
        if uniq_start_index is not None: 
            unique_idx = [uniq_start_index]
            sg1 = sg_seq_nx[uniq_start_index]
            for i in range(uniq_start_index, len(sg_seq_nx)):
                sg2 = sg_seq_nx[i]
                if nx.utils.misc.graphs_equal(sg1, sg2) == False:
                    sg1 = sg2
                    # if graph has nodes
                    if len(list(nx.get_node_attributes(sg2, "label").values())) > 0:
                        unique_idx.append(i)
            number_of_graphs.append(len(unique_idx))
        else:
            number_of_graphs.append(0)
            continue

    # convert to pyg.Data and embed
    all_pyg_graphs = []
    all_labels = []
    all_edge_labels = []
    
    for idx in unique_idx:
        labels = list(nx.get_node_attributes(sg_seq_nx[idx], "label").values())
        all_labels.extend(labels)
        edges = list(nx.get_edge_attributes(sg_seq_nx[idx], "label").values())
        all_edge_labels.extend(edges)

    node_embed = text2embedding(model, tokenizer, device, all_labels)
    edge_embed = text2embedding(model, tokenizer, device, all_edge_labels)

    node_idx_start, edge_idx_start = 0, 0
    for idx in unique_idx:
        pyg_graph = from_networkx(sg_seq_nx[idx])

        node_idx_end = len(pyg_graph.label)
        pyg_graph.x = node_embed[node_idx_start:node_idx_start + node_idx_end]
        node_idx_start += node_idx_end
        
        pe_transform = AddLaplacianEigenvectorPE(k=min(pyg_graph.x.shape[0] - 1, LPE_NUM), attr_name="laplacian_eigenvector_pe")
        pyg_graph = pe_transform(pyg_graph)
        pe = pyg_graph.laplacian_eigenvector_pe
        if pe.size(1) < LPE_NUM:
            num_missing = LPE_NUM - pe.size(1)
            pad = pe.new_zeros(pe.size(0), num_missing)
            pe = torch.cat([pe, pad], dim=1)
        pyg_graph.x = torch.cat([pyg_graph.x, pe], dim=-1) # n, 1024 + LPE_NUM

        if hasattr(pyg_graph, "edge_label"):
            edge_idx_end = len(pyg_graph.edge_label)
            pyg_graph.edge_attr = edge_embed[edge_idx_start:edge_idx_start + edge_idx_end]
            edge_idx_start += edge_idx_end
        else:
            pyg_graph.edge_attr = torch.zeros((0, 1024))
            pyg_graph.edge_label = []
        
        # we have to pad edge features too to match extended by LPE_NUM g.x
        pad = pyg_graph.edge_attr.new_zeros(pyg_graph.edge_attr.size(0), LPE_NUM)
        pyg_graph.edge_attr = torch.cat([pyg_graph.edge_attr, pad], dim=-1)

        all_pyg_graphs.append(pyg_graph)
    
    assert node_idx_start == len(node_embed)
    assert edge_idx_start == len(edge_embed)

    assert len(all_pyg_graphs) == len(unique_idx)
    int_names = [int(sg_seq_keys[idx].split("_")[1]) for idx in unique_idx] # int frame numbers from a to b
    assert int_names != 0
    norm_min, norm_max = min(int_names), max(int_names)
    if len(int_names) == 1:
        original_indices.append([0])
    else:
        original_indices.append([(i - norm_min) / (norm_max - norm_min) for i in int_names])

    torch.save(all_pyg_graphs, f"{root_path}/preprocessed_{MODEL_NAME}/graphs/pyg.pt")

    with open(f"{root_path}/preprocessed_{MODEL_NAME}/original_indices.pkl", 'wb') as f:
        pickle.dump(original_indices, f)

def preprocess_questions():
    with open(f"{root_path}/qa.json") as file:
        qa = json.load(file)
    questions = [i["question"] for i in qa]
    q_embed = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embed, f"{root_path}/preprocessed_{MODEL_NAME}/q_embed.pt")


if __name__ == "__main__":
    preprocess_graphs()
    preprocess_questions()
