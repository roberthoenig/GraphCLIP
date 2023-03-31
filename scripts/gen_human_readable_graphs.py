import json
from pprint import pprint
import torch

def get_human_readable_graph(d):
    id_to_idx = {}
    # TODO: deal with multiple object names?
    n_obj_nodes = len(d['objects'])
    obj_names = [obj['names'][0] for obj in d['objects']]
    attrs = []
    for o in d['objects']:
        attr_list = o.get('attributes', [])
        attrs.append(attr_list)
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx+1
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.long)
    edge_labels = []
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
        edge_labels.append(rel['predicate'])
    adj_list = edge_index.t().tolist()
    res = {
        "node labels": obj_names,
        "adjacency list": adj_list,
        "edge labels": edge_labels,
    }
    return res

def get_human_readable_graph_2(d):
    id_to_idx = {}
    # TODO: deal with multiple object names?
    n_obj_nodes = len(d['objects'])
    obj_names = [obj['names'][0] for obj in d['objects']]
    attrs = []
    for idx, o in enumerate(d['objects']):
        attr_list = o.get('attributes', [])
        attrs.append(attr_list)
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx+1
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.long)
    edge_labels = []
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
        edge_labels.append(rel['predicate'])
    adj_list = edge_index.t().tolist()
    adj_list_w_edge_labels = list(zip(adj_list, edge_labels))
    nodes = list(zip(range(1, n_obj_nodes+1), obj_names, attrs))
    res = {
        "node indices with node label and node attributes": nodes,
        "adjacency list with edge labels": adj_list_w_edge_labels,
    }
    return res

PROMPT = "Please parse the following labeled scene graph into an equivalent human-readable sentence. Please only include information that is explicitly stated in the graph. Your description should contain all information in the graph and is not limited in length. The ordering of the edges and nodes is arbitrary, so they are all equally important."

PROMPT2 = """
Please shorten your previous response such that it contains at most seventy words and retains as much information about the scene graph as possible. Write as terse as possible.
"""

def main():
    with open('datasets/visual_genome/raw/scene_graphs_small.json', 'r') as f:
        scene_graphs_dict = json.load(f)

    for d in scene_graphs_dict[:1]:
        print(PROMPT)
        print(get_human_readable_graph_2(d))
        print()
        print(PROMPT2)

if __name__ == '__main__':
    main()