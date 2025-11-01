import os 
import json
import pickle
from copy import deepcopy
from itertools import pairwise, chain

import torch
import networkx as nx
from tqdm import tqdm
from loguru import logger
from os.path import join as spj
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.utils.lm_modeling import load_model, load_text2embedding


root_path = os.environ["AGQA_ROOT"]
LPE_NUM = 4
logger.info(f"Setting lpe={LPE_NUM}")
with open(spj(root_path, "data", "ENG.txt"), "rt") as file:
    MAPPING = json.load(file)
### load embed model
MODEL_NAME = "mbert"
logger.info(f"Loading LM={MODEL_NAME}")
model, tokenizer, device = load_model[MODEL_NAME]()
logger.info(device)
text2embedding = load_text2embedding[MODEL_NAME]
SG_GLOBAL = dict() # contain in RAM low num of graphs to save i/o bound


def textualize_graph(nx_graph):
    description_nodes = []
    description_edges = []

    for idx, lab_ in dict(nx_graph.nodes(data=True)).items():
        description_nodes.append(f"{idx}: {lab_['label']}")
    description_nodes = ", ".join(description_nodes)

    for edg_ in nx_graph.edges(data=True):
        description_edges.append(f"{edg_[0]} {edg_[2]['label']} {edg_[1]}")
    description_edges = "; ".join(description_edges)
    
    return "Nodes:" + "\n" + description_nodes + "\n" + "Edges:" + "\n" + description_edges

def load_grounding_frames(grounding_item):
    # see more in AQGA README 'Scene graph grounding'
    # CORNER CASE 1: NO GROUNDING
    if not grounding_item:
        return []
    
    # CORNER CASE 2: only equal cases
    if all([key_pair.split("-")[0] == key_pair.split("-")[1] for key_pair in grounding_item.keys()]):
        return []

    frames_idx = set()
    for ground_elem in list(chain.from_iterable(grounding_item.values())):
        if ground_elem.split("/")[-1].startswith("0"):
            frames_idx.add(ground_elem.split("/")[-1])
    return list(frames_idx)

def parse_sg_keys(sg_item):
    seq = []
    frames = sorted([i for i in sg_item.keys() if i.startswith('0')], key=lambda x: int(x[-6:].lstrip('0')))
    for key_frame in frames:
        relevante_keys = [key for key in sg_item.keys() if key.endswith(key_frame)]

        G = nx.DiGraph()
        object_set = set() # there are no multi object cases
        object_set.add("o1") # action genome graph is always related to person

        for entity_key in [key for key in relevante_keys if key.startswith('o')]:
            object_set.add(sg_item[entity_key]["class"])
        # take objects also from relations and verbs
        for entity_key in [key for key in relevante_keys
                if (key.startswith('r') or key.startswith('v'))]:
            object_set.update([item["class"] for item in sg_item[entity_key]["objects"]])
        sg_object_mapping = {o_class: idx for idx, o_class in enumerate(list(object_set))}

        # add nodes
        for obj in object_set:
            G.add_node(sg_object_mapping[obj], label=MAPPING[obj])
        # add edges
        for entity_key in [key for key in relevante_keys
                if (key.startswith('r') or key.startswith('v'))]:
            edge_obj = sg_item[entity_key]["objects"]
            for e_obj in edge_obj:
                G.add_edge(sg_object_mapping["o1"], sg_object_mapping[e_obj["class"]],
                           label=MAPPING[sg_item[entity_key]["class"]])
        seq.append(G)
    return seq, frames

def preprocess_graphs():
    logger.info("working with graphs")

    os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/", exist_ok=True)
    
    for split in ["train", "test"]:
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs/", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs_networkx/", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/descs", exist_ok=True)

        # iterate over splits
        sg_file = f"AGQA_{split}_stsgs.pkl"
        with open(f"{root_path}/data/AGQA_scene_graphs/{sg_file}", "rb") as file:
            sg_data = pickle.load(file)

        for sg_name, sg_item in tqdm(sg_data.items()):
            sg_seq_nx, seq_frame_names = parse_sg_keys(sg_item)
            assert len(sg_seq_nx) > 0
            with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs_networkx/{sg_name}.pkl", "wb") as f:
                pickle.dump(sg_seq_nx, f)
            
            # leave only unique graphs and save indices
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
            assert len(unique_idx) > 0
            if (len(sg_seq_nx) - 1) not in unique_idx:
                unique_idx.append(len(sg_seq_nx) - 1) # to close range to the end

            # convert to pyg.Data and embed
            all_pyg_graphs = {}
            all_labels = []
            all_edge_labels = []
            
            for idx in unique_idx:
                labels = list(nx.get_node_attributes(sg_seq_nx[idx], "label").values())
                all_labels.extend(labels)
                edges = list(nx.get_edge_attributes(sg_seq_nx[idx], "label").values())
                all_edge_labels.extend(edges)

            node_embed = text2embedding(model, tokenizer, device, all_labels)
            edge_embed = text2embedding(model, tokenizer, device, all_edge_labels)
            
            pairwise_array = deepcopy(unique_idx)
            pairwise_array.append(len(sg_seq_nx) - 1) # duplicate last elem because right bound is not included
            node_idx_start, edge_idx_start = 0, 0
            description = {}
            for idx_l, idx_r in pairwise(pairwise_array):
                pyg_graph = from_networkx(sg_seq_nx[idx_l])
                description_local = textualize_graph(sg_seq_nx[idx_l])
                description[(seq_frame_names[idx_l], seq_frame_names[idx_r])] = description_local

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

                # left range include, right exclude (except last)
                all_pyg_graphs[(seq_frame_names[idx_l], seq_frame_names[idx_r])] = pyg_graph
                assert len(all_pyg_graphs) > 0
            
            assert node_idx_start == len(node_embed)
            assert edge_idx_start == len(edge_embed)

            SG_GLOBAL[sg_name] = all_pyg_graphs
            torch.save(all_pyg_graphs, f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs/{sg_name}.pt")
            with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/descs/{sg_name}.pkl", "wb") as f:
                pickle.dump(description, f)

def preprocess_qa():
    for split in ["train", "test"]:
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}", exist_ok=True)
        QA2SQ = {} # store mapping to grounded graphs in RAM instead of saving/loading to speed up

        logger.info(f"Loading QA")
        qa_data_path = f"{root_path}/data/AGQA_balanced/{split}_balanced.txt"
        qa_json = json.load(open(qa_data_path, mode='r', encoding='utf8'))

        for qa_key, qa_item in tqdm(qa_json.items()):
            seq_graph = SG_GLOBAL[qa_item['video_id']]

            grounding_idx = sorted(load_grounding_frames(qa_item["sg_grounding"]),
                key=lambda x: int(x[-6:].lstrip('0')))
            unique_g_idx = list()
            if grounding_idx:
                all_ranges = list(seq_graph.keys())
                for key_range in all_ranges:
                    for d_idx in grounding_idx:
                        if int(d_idx.lstrip()) in range(int(key_range[0].lstrip()), int(key_range[1].lstrip())):
                            unique_g_idx.append(key_range)
                # range will not work for the last range - manual check
                if grounding_idx[-1] == all_ranges[-1][0]:
                    unique_g_idx.append(all_ranges[-1])
            else:
                # quite often
                unique_g_idx = [key_range for key_range in list(seq_graph.keys())]

            QA2SQ[qa_key] = sorted(unique_g_idx, key=lambda x: int(x[0]))
            assert len(QA2SQ[qa_key]) > 0
        
        with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/qa2sg.pkl", 'wb') as f:
            pickle.dump(QA2SQ, f)

if __name__ == "__main__":
    # iterate over graphs and embed all keyframes
    # for AGQA (2M QA >> 10K SG) it is efficient
    preprocess_graphs()
    
    # because each QA contains it's own grounding frame set
    # we save unique SG idx for each element
    preprocess_qa()
