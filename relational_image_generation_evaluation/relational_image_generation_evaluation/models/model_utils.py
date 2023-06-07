import torch
from torch_geometric.data import Batch

def batch_transform(fn, batch, *args):
    data_list = batch.to_data_list()
    transformed_data_list = [fn(data, *args) for data in data_list]
    transformed_batch = Batch.from_data_list(transformed_data_list)
    return transformed_batch

def tokens_to_embeddings(data, embedding):
    num_embs = embedding.num_embeddings
    data.edge_attr[data.edge_attr < 0] = num_embs + data.edge_attr[data.edge_attr < 0]
    data.edge_attr = embedding(torch.clone(data.edge_attr)).reshape(data.edge_attr.shape[0], -1)
    data.x[data.x < 0] += num_embs
    data.x = embedding(torch.clone(data.x)).reshape(data.x.shape[0], -1)
    return data

def tokens_to_embeddings_batched(batch, embeddings):
    return batch_transform(tokens_to_embeddings, batch, embeddings)

def get_master_node_indices(batch):
    _, counts = torch.unique(batch, return_counts=True)
    master_node_indices = torch.cumsum(counts, dim=0) - 1
    return master_node_indices

# x: (n_nodes x n_batches, n_node_features)
# batch: (n_nodes x n_batches)
def global_master_pool(x, batch):
    out = x[get_master_node_indices(batch)]
    return out