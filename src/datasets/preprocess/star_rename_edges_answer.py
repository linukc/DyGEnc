import os 
import json
import pickle
import random

import spacy
import torch
import networkx as nx
from tqdm import tqdm
from loguru import logger
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.utils.lm_modeling import load_model, load_text2embedding

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # если используешь GPU:
    torch.cuda.manual_seed_all(seed)
    # для повторяемости на cudnn (может замедлить):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

verbs_synonyms = {
    "close": ["shut", "seal", "lock", "fasten", "bar"],
    "drink": ["sip", "gulp", "imbibe", "quaff", "consume"],
    "eat": ["consume", "devour", "ingest", "munch", "chew"],
    "grasp": ["grab", "clutch", "seize", "hold", "grip"],
    "hold": ["grip", "carry", "clutch", "retain", "support"],
    "lie": ["recline", "rest", "stretch", "sprawl", "repose"],
    "open": ["unseal", "uncover", "unlock", "spread", "expose"],
    "put": ["place", "set", "position", "lay", "deposit"],
    "sit": ["perch", "rest", "lounge", "settle", "recline"],
    "stand": ["rise", "upright", "erect", "remain", "hold"],
    "take": ["grab", "seize", "pick", "collect", "acquire"],
    "throw": ["toss", "hurl", "fling", "pitch", "lob"],
    "tidy": ["clean", "organize", "arrange", "neaten", "straighten"],
    "turn": ["rotate", "spin", "twist", "flip", "pivot"],
    "walk": ["stroll", "amble", "stride", "march", "saunter"],
    "wash": ["cleanse", "scrub", "rinse", "bathe", "soak"],
    "watch": ["observe", "view", "look", "monitor", "see"],
}



EXP_VARIANT = "rename_edges_answer"
root_path = os.environ["STAR_ROOT"]
LPE_NUM = 4
logger.info(f"Setting lpe={LPE_NUM}")
### load embed model
MODEL_NAME = "mbert"
logger.info(f"Loading LM={MODEL_NAME}")
model, tokenizer, device = load_model[MODEL_NAME]()
logger.info(device)
text2embedding = load_text2embedding[MODEL_NAME]


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

def preprocess_graphs(seed, DROP_RATE):
    logger.info("working with graphs")
    n2txt = open(f"{root_path}/data/mapping_annotations/object_classes.txt").readlines()
    nodeid2text = {elem.split()[0]: elem.split()[1] for elem in n2txt}
    text2nodeid = {elem.split()[1]: elem.split()[0] for elem in n2txt}
    r2txt = open(f"{root_path}/data/mapping_annotations/relationship_classes.txt").readlines()
    relid2text = {elem.split()[0]: elem.split()[1] for elem in r2txt}
    text2relid = {elem.split()[1]: elem.split()[0] for elem in r2txt}
    a2v_o = open(f"{root_path}/data/mapping_annotations/action_mapping.txt").readlines()
    aid2vid_oid = {elem.split()[0]: (elem.split()[1], elem.split()[2]) for elem in a2v_o}
    v2txt = open(f"{root_path}/data/mapping_annotations/verb_classes.txt").readlines()
    verbid2text = {elem.split()[0]: elem.split()[1] for elem in v2txt}
    text2verbid = {elem.split()[1]: elem.split()[0] for elem in v2txt}
    
    os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/", exist_ok=True)
    
    for split in ["val"]:
        number_of_graphs = []
        original_indices = []
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}", exist_ok=True)
        logger.info(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}")
        json_file = f"STAR_{split}.json"

        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/graphs", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/descs", exist_ok=True)

        ### debug set
        debug_answer_set = set()
        # iterate over splits
        with open(f"{root_path}/data/{json_file}") as file:
            json_data = json.load(file)
            nlp = spacy.load("en_core_web_sm")
            
            # iterate over questions
            for item_idx, item in enumerate(tqdm(json_data)):

                ### Edge name to drop
                edge_name_to_drop = "not_found"
                # analyze showes it is from verb mapping and ("lying_on", "eating", "sitting_on") [in verb it is lie, eat, sit]
                ### Case1 
                if not item["answer"].startswith("The"):
                    #case1 - only verf
                    if not "the" in item["answer"].lower():
                        edge_name_to_drop = item["answer"].lower().rstrip(".")
                        edge_name_to_drop = nlp(edge_name_to_drop)[0].lemma_ # first is a word, last is a possible preposition
                    #case2 - verb + obj
                    else:
                        edge_name_to_drop, _ = item["answer"].split(" the ")
                        edge_name_to_drop = edge_name_to_drop.lower().strip()
                        edge_name_to_drop = nlp(edge_name_to_drop)[0].lemma_ # first is a word, last is a possible preposition

                NEW_EDGE = None
                if edge_name_to_drop != "not_found":
                    syn_edges = verbs_synonyms.get(edge_name_to_drop)
                    if syn_edges:
                        NEW_EDGE = random.choice(syn_edges)
                    else:
                        logger.warning(f"can't find synonym for {edge_name_to_drop}")
                if edge_name_to_drop == "not_found":
                    debug_answer_set.add(item["answer"])              
                edges_to_drop_list_of_tuples = []

                sg_seq_keys = sorted(list(map(int, item["situations"].keys())))
                sg_seq = [item["situations"][f"{key}".zfill(6)] for key in sg_seq_keys]
                sg_seq_nx = []

                # iterate over individual G in the sequence of graphs at the particular event (question)
                for sg in sg_seq:
                    G = nx.DiGraph()

                    actions = sg["actions"]
                    obj_set = set()
                    for rp in sg["rel_pairs"]:
                        obj_set.update(rp)
                    obj_set.update(sg["bbox_labels"])
                    # need to add person in case action is annotated but there is no node in bbox_labels or rel_pairs
                    obj_set.add("o000") # person node
                    # need to add all targeted objects in actions for the same reason
                    obj_set.update([aid2vid_oid[a][1] for a in actions])
                    # print(answer_id in obj_set)

                    inner_obj_mapping = {v: i for i, v in enumerate(list(obj_set))} # o000: 0, o0001: 1 ...
                    for k, v in inner_obj_mapping.items():
                        G.add_node(v, label=nodeid2text[k])

                    # handle that one edge can have multiple values
                    edges = {}
                    for rp, rl in zip(sg["rel_pairs"], sg["rel_labels"]):
                        a, b = inner_obj_mapping[rp[0]], inner_obj_mapping[rp[1]]
                        if f"{a}->{b}" in edges:
                            edges[f"{a}->{b}"].append(relid2text[rl])
                        else:
                            edges[f"{a}->{b}"] = [relid2text[rl]]
                    
                    # extend with verbs
                    for act in actions:
                        # action is always directed edge from person to an object
                        a, b = inner_obj_mapping["o000"], inner_obj_mapping[aid2vid_oid[act][1]]
                        if f"{a}->{b}" in edges:
                            edges[f"{a}->{b}"].append(verbid2text[aid2vid_oid[act][0]])
                        else:
                            edges[f"{a}->{b}"] = [verbid2text[aid2vid_oid[act][0]]]

                    local_drop_edges_list = []
                    for k, v in edges.items():
                        a, b = map(int, k.split("->"))
                        G.add_edge(a, b, label=", ".join(v))
                        for elem in v:
                            # ("lying_on", "eating", "sitting_on") [in verb it is lie, eat, sit]
                            if elem == edge_name_to_drop or \
                            (edge_name_to_drop == "eat" and elem == "eating") or \
                            (edge_name_to_drop == "lie" and elem == "lying_on") or \
                            (edge_name_to_drop == "sit" and elem == "sitting_on"):
                                local_drop_edges_list.append((a, b, elem))
                    
                    sg_seq_nx.append(G)
                    edges_to_drop_list_of_tuples.append(local_drop_edges_list)

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

                ### drop answer nodes (edges will drop automatically after node removal)
                # calculate gt node num
                to_remove_idx = []
                for idx in unique_idx:
                    local_drop_edges_list = edges_to_drop_list_of_tuples[idx]
                    if local_drop_edges_list: #and answer_node_label in sg_seq_nx[idx]
                        to_remove_idx.append(idx)
                random_idx_to_operate = random.sample(to_remove_idx, int(len(to_remove_idx) * DROP_RATE/ 3))
                debug_counter1, debug_counter2 = 0, 0
                for idx in random_idx_to_operate:
                    graph = sg_seq_nx[idx]
                    # do not forget that i do not delete full edge if it has multiple values
                    # assert len(edges_to_drop_list_of_tuples[idx]) == 1
                    # print(edges_to_drop_list_of_tuples[idx])
                    for pair in edges_to_drop_list_of_tuples[idx]:
                        edge = graph[pair[0]][pair[1]]['label']
                        assert NEW_EDGE != None
                        new_edge = edge.replace(pair[2], NEW_EDGE)
                        graph[pair[0]][pair[1]]['label'] = new_edge
                
                # convert to pyg.Data and embed
                all_pyg_graphs = []
                all_labels = []
                all_edge_labels = []
                description = []
                
                for idx in unique_idx:
                    labels = list(nx.get_node_attributes(sg_seq_nx[idx], "label").values())
                    all_labels.extend(labels)
                    edges = list(nx.get_edge_attributes(sg_seq_nx[idx], "label").values())
                    all_edge_labels.extend(edges)

                    description_local = textualize_graph(sg_seq_nx[idx])
                    description_local = f"Start of graph {idx}\n{description_local}\nEnd of graph {idx}"
                    description.append(description_local)

                with open(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/descs/{item_idx}.pkl", "wb") as f:
                    pickle.dump("\n".join(description), f)
                    
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
                int_names = [int(sg_seq_keys[idx]) for idx in unique_idx] # int frame numbers from a to b
                assert int_names != 0
                norm_min, norm_max = min(int_names), max(int_names)
                if len(int_names) == 1:
                    original_indices.append([0])
                else:
                    original_indices.append([(i - norm_min) / (norm_max - norm_min) for i in int_names])
    
                torch.save(all_pyg_graphs, f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/graphs/{item_idx}.pt")
    
        with open(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/num_graphs_metadata.pkl", 'wb') as f:
            pickle.dump(number_of_graphs, f)
        with open(f"{root_path}/preprocessed_{MODEL_NAME}_{EXP_VARIANT}_{DROP_RATE}_{seed}/{split}/original_indices.pkl", 'wb') as f:
            pickle.dump(original_indices, f)

if __name__ == "__main__":
    for drop_rate in [1, 2, 3]:
        for seed in [0, 1, 2]:
            set_seed(seed)
            preprocess_graphs(seed, drop_rate)