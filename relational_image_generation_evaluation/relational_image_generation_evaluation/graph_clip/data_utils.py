import torch
from torch_geometric.data import Data


def networkx_to_dict(nx):
    objects = []
    for node_id in nx.nodes:
        obj = {
            'names': [nx.nodes[node_id]['name']],
            'attributes': nx.nodes[node_id]['attributes'],
            'object_id': node_id,
        }
        objects.append(obj)
    relationships = []
    for (subject_id, object_id) in nx.edges:
        rel = {
            "object_id": object_id,
            "subject_id": subject_id,
            "predicate": nx.edges[(subject_id, object_id)]['predicate'],
        }
        relationships.append(rel)
    d = {
        'objects': objects,
        'relationships': relationships,
    }
    return d
    

def dict_to_pyg_graph(d, txt_enc,  use_long_rel_enc):
    # x: [num_nodes, num_txt_features]
    id_to_idx = {}
    # TODO: deal with multiple object names?
    n_obj_nodes = len(d['objects'])
    x = txt_enc([obj['names'][0] for obj in d['objects']])
    attrs = []
    attr_to_x = []
    for idx, o in enumerate(d['objects']):
        for attr in o.get('attributes', []):
            attrs.append(attr)
            attr_to_x.append(idx)
    n_attrs = len(attrs)
    if n_attrs == 0:
        attrs = torch.zeros((0, 2), dtype=torch.int64)
    else:
        attrs = txt_enc(attrs)
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.int64)
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
    attrs_edge_index = torch.zeros((2, n_attrs), dtype=torch.int64)
    for attr_idx, x_idx in enumerate(attr_to_x):
        attrs_edge_index[:, attr_idx] = torch.tensor([attr_idx+n_obj_nodes, x_idx])
    # edge_attr: [num_edges, num_txt_features]
    if len(d['relationships']) == 0:
        edge_attr = torch.zeros((0, 2), dtype=torch.int64)
    else:
        rel_txts = []
        for rel in d['relationships']:
            if use_long_rel_enc:
                subj_txt = d['objects'][id_to_idx[rel['subject_id']]]['names'][0]
                obj_txt = d['objects'][id_to_idx[rel['object_id']]]['names'][0]
                rel_txt = rel['predicate']
                compound_txt = " ".join([subj_txt, rel_txt, obj_txt])
            else:
                compound_txt = rel['predicate']
            rel_txts.append(compound_txt)
        edge_attr = txt_enc(rel_txts)
    attrs_edge_attr = -3*torch.ones((n_attrs, 2), dtype=torch.int64)
    
    data = Data(x=torch.cat([x, attrs]),
        edge_attr=torch.cat([edge_attr, attrs_edge_attr]),
        edge_index=torch.cat([edge_index, attrs_edge_index], dim=1),
        obj_nodes=torch.arange(0, n_obj_nodes),
        attr_nodes=torch.arange(n_obj_nodes, n_obj_nodes + n_attrs),
    )
    return data


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