import torch

# x: (n_nodes x n_batches, n_node_features)
# batch: (n_nodes x n_batches)
def global_master_pool(x, batch):
    _, counts = torch.unique(batch, return_counts=True)
    last_occurrence_indices = torch.cumsum(counts, dim=0) - 1
    out = x[last_occurrence_indices]
    return out