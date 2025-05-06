from torch_geometric.data import Batch


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graphs' in batch:
        all_data = []
        seq_lengths = []
        for seq in batch['graphs']:
            all_data.extend(seq)
            seq_lengths.append(len(seq))

        batch['graphs'] = Batch.from_data_list(all_data, 
            exclude_keys=["edge_label", "label", "laplacian_eigenvector_pe"])
        batch['seq_lengths'] = seq_lengths
    return batch
