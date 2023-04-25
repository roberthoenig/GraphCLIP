import torch
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

def get_master_node_indices(batch):
    _, counts = torch.unique(batch, return_counts=True)
    master_node_indices = torch.cumsum(counts, dim=0) - 1
    return master_node_indices

# x: (n_nodes x n_batches, n_node_features)
# batch: (n_nodes x n_batches)
def global_master_pool(x, batch):
    out = x[get_master_node_indices(batch)]
    return out

def dropout_node_keep_master_nodes(edge_index, batch, p = 0.5,
                 num_nodes = None,
                 training = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    node_mask[get_master_node_indices(batch)] = True
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)