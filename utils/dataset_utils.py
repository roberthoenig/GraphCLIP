from re import T
import zipfile
from torch.utils import data
from torch_geometric.data import Data, Batch
import torch
from tqdm import tqdm
import logging
import random

def unzip_file(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def is_not_edgeless(data):
    return data.edge_index.shape[1] > 0      

def in_mscoco_val(data):
    return data.in_coco_val.item()

def dataset_filter(dataset, filter=None):
    # Filter
    if filter == "remove_edgeless_graphs":
        filter_fn = is_not_edgeless
    elif filter == "remove_mscoco_val":
        filter_fn = lambda x: not in_mscoco_val(x)
    elif filter == "keep_mscoco_val":
        filter_fn = in_mscoco_val
    elif filter is None:
        filter_fn = lambda x: True
    else:
        raise Exception(f"Unknown filter {filter}")
    logging.info("Filtering dataset...")
    filtered_indexes = [i for i in tqdm(range(len(dataset))) if filter_fn(dataset[i])]
    filtered_dataset = data.Subset(dataset, filtered_indexes)
    return filtered_dataset

def add_master_node_with_bidirectional_edges(data):
    n_nodes = data.x.shape[0]
    new_node = -4*torch.ones((1, 2), dtype=torch.int64)
    x = torch.cat([data.x, new_node])
    new_edges = torch.tensor([[n_nodes, t] for t in data.obj_nodes.tolist()] + [[t, n_nodes] for t in data.obj_nodes.tolist()], dtype=torch.int).t()
    edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    edge_attr = torch.cat([
        data.edge_attr,
        -1*torch.ones((n_nodes, 2), dtype=torch.int64),
        -2*torch.ones((n_nodes, 2), dtype=torch.int64)
    ])
    d = {'x': x,
         'edge_index': edge_index,
         'edge_attr': edge_attr
    }
    d_orig = data.to_dict()
    d_orig.update(d)
    data_with_master_node = Data.from_dict(d_orig)
    return data_with_master_node

def add_master_node_with_incoming_edges(data):
    n_nodes = data.x.shape[0]
    new_node = torch.ones(1, 2, dtype=torch.float32)
    x = torch.cat([data.x, new_node])
    new_edges = torch.tensor([[t, n_nodes] for t in data.obj_nodes.tolist()], dtype=torch.int).t()
    edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    edge_attr = torch.cat([data.edge_attr, -1*torch.ones(n_nodes, 2)])
    d = {'x': x,
         'edge_index': edge_index,
         'edge_attr': edge_attr
    }
    d_orig = data.to_dict()
    d_orig.update(d)
    data_with_master_node = Data.from_dict(d_orig)
    return data_with_master_node

# !!! Assumes that each attribute has exactly ONE outgoing edge !!!
def transfer_attributes(data):
    if len(data.attr_nodes) == 0:
        return data
    attr_to_transfer = random.sample(data.attr_nodes.tolist(), 1)[0]
    node_to_receive = random.sample(data.obj_nodes.tolist(), 1)[0]
    assert len((data.edge_index[0] == attr_to_transfer).nonzero()) == 1
    edge_idx = (data.edge_index[0] == attr_to_transfer).nonzero()[0].item()
    orig_node = data.edge_index[1, edge_idx]
    # This shouldn't happen too often
    if orig_node == node_to_receive:
        return data
    data.edge_index[1, edge_idx] = node_to_receive
    return data

def transfer_attributes_batched(batch):
    return batch_transform(transfer_attributes, batch)

def batch_transform(fn, batch, *args):
    data_list = batch.to_data_list()
    transformed_data_list = [fn(data, *args) for data in data_list]
    transformed_batch = Batch.from_data_list(transformed_data_list)
    return transformed_batch

def tokens_to_embeddings(data, embedding):
    num_embs = embedding.num_embeddings
    data.edge_attr[data.edge_attr < 0] = num_embs + data.edge_attr[data.edge_attr < 0]
    data.edge_attr = embedding(data.edge_attr).reshape(data.edge_attr.shape[0], -1)
    data.x[data.x < 0] += num_embs
    data.x = embedding(torch.clone(data.x)).reshape(data.x.shape[0], -1)
    return data


def tokens_to_embeddings_batched(batch, embeddings):
    return batch_transform(tokens_to_embeddings, batch, embeddings)
