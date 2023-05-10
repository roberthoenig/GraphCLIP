from re import T
import zipfile
from torch.utils import data
from torch_geometric.data import Data, Batch
import torch
from tqdm import tqdm
import logging
import random
import json
import os.path as osp
import numpy as np

def unzip_file(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def is_not_edgeless(data):
    return data.edge_index.shape[1] > 0      

def in_mscoco_val(data):
    return data.in_coco_val.item()

def make_is_not_visualgenome_duplicate(vg_dupes):
    def is_not_visualgenome_duplicate(data):
        return data['image_id'].item() == vg_dupes[str(data['coco_id'].item())][0]
    return is_not_visualgenome_duplicate

def make_remove_adv_dataset_samples():
    with open("datasets/visual_genome/raw/realistic_adversarial_samples.json", "r") as f:
        samples = json.load(f)
        ids = [int(id) for id in samples.keys()]
    def remove_adv_dataset_samples(data):
        return not (data.image_id.item() in ids)
    return remove_adv_dataset_samples

def dataset_filter(dataset, filters=[]):
    filter_fns = []
    for filter in filters:
        # Filter
        if filter == "remove_edgeless_graphs":
            filter_fn = is_not_edgeless
        elif filter == "remove_mscoco_val":
            filter_fn = lambda x: not in_mscoco_val(x)
        elif filter == "keep_mscoco_val":
            filter_fn = in_mscoco_val
        elif filter == "remove_visualgenome_duplicates":
            with open("datasets/visual_genome/raw/visualgenome_duplicates.json", "r") as f:
                vg_dupes = json.load(f)
            filter_fn = make_is_not_visualgenome_duplicate(vg_dupes)
        elif filter == "remove_adv_dataset_samples":
            filter_fn = make_remove_adv_dataset_samples()
        elif filter is None:
            filter_fn = lambda x: True
        else:
            raise Exception(f"Unknown filter {filter}")
        filter_fns.append(filter_fn)
    logging.info(f"Filtering dataset...")
    filtered_indexes = [i for i in tqdm(range(len(dataset))) if all(f(dataset[i]) for f in filter_fns)]
    dataset = data.Subset(dataset, filtered_indexes)
    return dataset

def add_master_node_with_bidirectional_edges(data):
    n_nodes = data.x.shape[0]
    n_obj_nodes = len(data.obj_nodes)
    new_node = -4*torch.ones((1, 2), dtype=torch.int64)
    x = torch.cat([data.x, new_node])
    new_edges = torch.tensor([[n_nodes, t] for t in data.obj_nodes.tolist()] + [[t, n_nodes] for t in data.obj_nodes.tolist()], dtype=torch.int).t()
    edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    edge_attr = torch.cat([
        data.edge_attr,
        -1*torch.ones((n_obj_nodes, 2), dtype=torch.int64),
        -2*torch.ones((n_obj_nodes, 2), dtype=torch.int64)
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

with open("datasets/visual_genome/raw/relation_distribution.json", "r") as f:
    rel_distr = json.load(f)
SZ = 1_000_000
rel_replacements = np.random.choice(rel_distr["words"], size=SZ, p=rel_distr["probs"])
rel_ctr = 0

def sample_relation(data, txt_enc, replacement_prob):
    global rel_ctr
    global rel_replacements
    replace_mask = torch.rand(len(data.edge_attr)) <= replacement_prob
    replace_mask[(data.edge_attr < 0).all(dim=1)] = False
    n_replacements = replace_mask.int().sum()
    if rel_ctr + n_replacements >= SZ:
        rel_replacements = np.random.choice(rel_distr["words"], size=SZ, p=rel_distr["probs"])
        rel_ctr = 0
    replacements = rel_replacements[rel_ctr:rel_ctr+n_replacements]
    rel_ctr += n_replacements
    replacement_tokens = txt_enc(replacements)
    data.edge_attr[replace_mask] = replacement_tokens
    data.adv_affected_nodes = data.edge_index[:, replace_mask].flatten()
    return data

def transfer_attributes_batched(batch):
    return batch_transform(transfer_attributes, batch)

def make_sample_relation_batched(txt_enc, replacement_prob=1.0):
    def sample_relation_batched(batch):
        return batch_transform(sample_relation, batch, txt_enc, replacement_prob)
    return sample_relation_batched

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

def process_adversarial_dataset(in_dir, in_fname):
    with open(osp.join(in_dir, in_fname), 'r') as f:
        adv_data = json.load(f)
    data_gt = []
    data_adv = []
    for v in adv_data:
        for target in ["gt", "adv"]:
            image_id = v["image_id"]
            object_id = v["changed_edge_obj"]
            subject_id = v["changed_edge_subj"]
            predicate = v["original_predicate" if target == "gt" else "adv_predicate"]
            objects = [
                {
                    'names': [v['subj_name']],
                    'object_id': subject_id,
                },
                {
                    'names': [v['obj_name']],
                    'object_id': object_id,
                },
            ]
            relationships = [{
                'object_id': object_id,
                'subject_id': subject_id,
                'predicate': predicate,
            }]
            sample = {
                'image_id': image_id,
                'objects': objects,
                'relationships': relationships,
            }
            if target == "gt":
                data_gt.append(sample)
            else:
                data_adv.append(sample)
        
    with open(osp.join(in_dir, "scene_graphs_gt.json"), 'w') as f:
        json.dump(data_gt, f)

    with open(osp.join(in_dir, "scene_graphs_adv.json"), 'w') as f:
        json.dump(data_adv, f)