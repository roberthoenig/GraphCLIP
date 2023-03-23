from re import T
import zipfile
from torch.utils import data
from torch_geometric.data import Data
import torch

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
    filtered_indexes = [i for i in range(len(dataset)) if filter_fn(dataset[i])]
    filtered_dataset = data.Subset(dataset, filtered_indexes)
    return filtered_dataset

def add_master_node(data):
    n_node_features = data.x.shape[1]
    n_nodes = data.x.shape[0]
    new_node = torch.ones(1, n_node_features, dtype=torch.float32)
    x = torch.cat([data.x, new_node])
    new_edges = torch.tensor([[n_nodes, t] for t in range(n_nodes)] + [[t, n_nodes] for t in range(n_nodes)], dtype=torch.int).t()
    edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    n_edge_features = data.edge_attr.shape[1]
    edge_attr = torch.cat([data.edge_attr, torch.ones(n_nodes, n_edge_features), torch.sin(torch.arange(0, n_edge_features).repeat(n_nodes, 1))])
    d = {'x': x,
         'edge_index': edge_index,
         'edge_attr': edge_attr
    }
    d_orig = data.to_dict()
    d_orig.update(d)
    data_with_master_node = Data.from_dict(d_orig)
    return data_with_master_node