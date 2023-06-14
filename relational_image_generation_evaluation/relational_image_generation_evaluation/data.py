from .download_weights import download_filtered_graphs, download_mscoco_graphs, download_adv_datasets
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import json

FILTERED_OBJECTS = ['man', 'person', 'window', 'tree', 'building', 'shirt', 'wall', 'woman', 'sign', 'sky', 'ground', 'grass', 'table', 'pole', 'head', 'light', 'water', 'car', 'hand', 'hair', 'people', 'leg', 'trees', 'clouds', 'ear', 'plate', 'leaves', 'fence', 'door', 'pants', 'eye', 'train', 'chair', 'floor', 'road', 'street', 'hat', 'snow', 'wheel', 'shadow', 'jacket', 'nose', 'boy', 'line', 'shoe', 'clock', 'sidewalk', 'boat', 'tail', 'cloud', 'handle', 'letter', 'girl', 'leaf', 'horse', 'bus', 'helmet', 'bird', 'giraffe', 'field', 'plane', 'flower', 'elephant', 'umbrella', 'dog', 'shorts', 'arm', 'zebra', 'face', 'windows', 'sheep', 'glass', 'bag', 'cow', 'bench', 'cat', 'food', 'bottle', 'rock', 'tile', 'kite', 'tire', 'post', 'number', 'stripe', 'surfboard', 'truck', 'logo', 'glasses', 'roof', 'skateboard', 'motorcycle', 'picture', 'flowers', 'bear', 'player', 'foot', 'bowl', 'mirror', 'background', 'pizza', 'bike', 'shoes', 'spot', 'tracks', 'pillow', 'shelf', 'cap', 'mouth', 'box', 'jeans', 'dirt', 'lights', 'legs', 'house', 'part', 'trunk', 'banana', 'top', 'plant', 'cup', 'counter', 'board', 'bed', 'wave', 'bush', 'ball', 'sink', 'button', 'lamp', 'beach', 'brick', 'flag', 'neck', 'sand', 'vase', 'writing', 'wing', 'paper', 'seat', 'lines', 'reflection', 'coat', 'child', 'toilet', 'laptop', 'airplane', 'letters', 'glove', 'vehicle', 'phone', 'book', 'branch', 'sunglasses', 'edge', 'cake', 'desk', 'rocks', 'frisbee', 'tie', 'tower', 'animal', 'hill', 'mountain', 'headlight', 'ceiling', 'cabinet', 'eyes', 'stripes', 'wheels', 'lady', 'ocean', 'racket', 'container', 'skier', 'keyboard', 'towel', 'frame', 'windshield', 'hands', 'back', 'track', 'bat', 'finger', 'pot', 'orange', 'fork', 'waves', 'design', 'feet', 'basket', 'fruit', 'broccoli', 'engine', 'guy', 'knife', 'couch', 'railing', 'collar', 'cars']
FILTERED_RELATIONSHIPS = ['on', 'has', 'in', 'of', 'wearing', 'with', 'behind', 'holding', 'on a', 'near', 'on top of', 'next to', 'has a', 'under', 'of a', 'by', 'above', 'wears', 'in front of', 'sitting on', 'on side of', 'attached to', 'wearing a', 'in a', 'over', 'are on', 'at', 'for', 'around', 'beside', 'standing on', 'riding', 'standing in', 'inside', 'have', 'hanging on', 'walking on', 'on front of', 'are in', 'hanging from', 'carrying', 'holds', 'covering', 'belonging to', 'between', 'along', 'eating', 'and', 'sitting in', 'watching', 'below', 'painted on', 'laying on', 'against', 'playing', 'from', 'inside of', 'looking at', 'with a', 'parked on', 'to', 'has an', 'made of', 'covered in', 'mounted on', 'says', 'growing on', 'across', 'part of', 'on back of', 'flying in', 'outside', 'lying on', 'worn by', 'walking in', 'sitting at', 'printed on', 'underneath', 'crossing', 'beneath', 'full of', 'using', 'filled with', 'hanging in', 'covered with', 'built into', 'standing next to', 'adorning', 'a', 'in middle of', 'flying', 'supporting', 'touching', 'next', 'swinging', 'pulling', 'growing in', 'sitting on top of', 'standing', 'lying on top of']
FILTERED_ATTRIBUTES = ['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'wooden', 'gray', 'silver', 'metal', 'orange', 'grey', 'tall', 'long', 'dark', 'pink', 'clear', 'standing', 'round', 'tan', 'glass', 'here', 'wood', 'open', 'purple', 'big', 'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young', 'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy', 'colorful', 'one', 'beige', 'bare', 'wet', 'light', 'square', 'little', 'closed', 'stone', 'blonde', 'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted', 'thick', 'part', 'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular', 'looking', 'grassy', 'dry', 'light brown', 'cement', 'leafy', 'wearing', 'tiled', "man's", 'light blue', 'baseball', 'cooked', 'pictured', 'curved', 'decorative', 'dead', 'eating', 'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt', 'watching', 'colored', 'stuffed', 'circular']

filtered_graphs=None

def load_filtered_graphs(testonly=False):
    if testonly:
        download_filtered_graphs()
        print('Loading filtered test graphs...')
        fg = torch.load(os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs_test_small.pt'))
        for i in range(len(fg)):
            fg[i].caption = get_caption(fg[i])
        print('Finished loading filtered test graphs')
        return fg
    global filtered_graphs
    if filtered_graphs is None:
        download_filtered_graphs()
        print('Loading filtered graphs...')
        filtered_graphs = torch.load(os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs.pt'))
        for i in range(len(filtered_graphs)):
            filtered_graphs[i].caption = get_caption(filtered_graphs[i])
        print('Finished loading filtered graphs')
        return filtered_graphs
    else:
        print('Using cached filtered graphs')
        return filtered_graphs

mscoco_graphs = None

def load_mscoco_graphs(testonly=False):
    if testonly:
        download_mscoco_graphs()
        print('Loading MSCOCO test graphs...')
        fg = torch.load(os.path.join(os.path.dirname(__file__), 'data', 'mscoco_graphs_test_small.pt'))
        print('Finished loading mscoco test graphs')
        return fg
    global mscoco_graphs
    if mscoco_graphs is None:
        download_mscoco_graphs()
        print('Loading mscoco graphs...')
        mscoco_graphs = torch.load(os.path.join(os.path.dirname(__file__), 'data', 'mscoco_graphs.pt'))
        print('Finished loading mscoco graphs')
        return mscoco_graphs
    else:
        print('Using cached filtered graphs')
        return mscoco_graphs
    
def copy_graph(g, nodes_restrict=None, edges_restrict=None):
    '''returns a copy of the graph g'''
    g_copy = nx.DiGraph()
    g_copy.add_nodes_from(g.nodes(data=True))
    g_copy.add_edges_from(g.edges(data=True))
    g_copy.labels = g.labels
    g_copy.image_id = g.image_id
    try:
        g_copy.image_w = g.image_w
        g_copy.image_h = g.image_h
    except:
        pass
    if nodes_restrict is not None:
        g_copy.remove_nodes_from([n for n in g_copy if n not in nodes_restrict])
    if edges_restrict is not None:
        g_copy.remove_edges_from([e for e in g_copy.edges() if e not in edges_restrict])
    g_copy.caption = get_caption(g_copy) # update the caption, which might be different due to the removal of nodes and edges
    return g_copy

def get_caption(graph):
    caption = ""
    entity_id_to_txt = {node: (', '.join(graph.nodes[node]['attributes']) + " "+ graph.nodes[node]['name']).strip() for node in graph.nodes}
    for edge in graph.edges:
        rel = graph.edges[edge]['predicate']
        subj_txt = entity_id_to_txt[edge[0]]
        obj_txt = entity_id_to_txt[edge[1]]
        caption += subj_txt + " " + rel + " " + obj_txt + ". "
    # for all nodes that are not occuring in any edge
    for node in graph.nodes:
        if node not in [edge[0] for edge in graph.edges] and node not in [edge[1] for edge in graph.edges]:
            caption += entity_id_to_txt[node] + ". "
    return caption.strip()

def plot_graph(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    max_y = max([y for x,y in pos.values()])
    n_nodes_top = len([n for n in g.nodes if pos[n][1] == max_y])
    longest_label = max([len(g.labels[n]) for n in g.nodes])
    plt.figure(figsize=(max(n_nodes_top*longest_label/10,15),5))
    nx.draw(g,pos=pos,labels=g.labels, with_labels=True, node_size=10, node_color="lightgray", font_size=8)
    nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=nx.get_edge_attributes(g,'predicate'),font_size=8)
    plt.show()

def get_all_undir_paths(g_dir, length=2):
    '''
    returns all undirected paths of length length in g_dir
    ignores loops
    gives back correct direction of edges
    '''
    paths = [[n] for n in g_dir.nodes]
    g_undir = g_dir.to_undirected()
    for _ in range(length):
        new_paths = []
        for path in paths:
            for n in g_undir.neighbors(path[-1]):
                if n not in path: # the graphs actually have cycles, so we need to check for that
                    new_paths.append(path+[n])
        paths = new_paths
    # make paths unique
    paths = [[(n1,n2) if (n1,n2) in g_dir.edges else (n2,n1) for n1,n2 in zip(path[:-1], path[1:])] for path in paths]
    paths = [tuple(sorted(path)) for path in paths]
    paths = list(set(paths))
    for path in paths:
        assert len(path) == length, f'path {path} has length {len(path)}'
    return paths

class OneEdgeDataset(Dataset):
    def __init__(self, testonly=False):
        '''
        Dataset of single edge graphs
        '''
        super().__init__()
        self.graphs = load_filtered_graphs(testonly=testonly)
        print('Generating one edge graphs...')
        self.one_edge_graphs = self.generate_one_edge_graphs()
        print('Finished generating one edge graphs')

    def generate_one_edge_graphs(self):
        one_edge_graphs = []
        for graph in tqdm(self.graphs):
            paths = get_all_undir_paths(graph, length=1)
            for path in paths:
                assert len(path) == 1, f'path {path} has length {len(path)}, should be 1'
                edge = path[0]
                one_edge_graph = copy_graph(graph, nodes_restrict=[edge[0], edge[1]], edges_restrict=[edge])
                one_edge_graphs.append(one_edge_graph)
                assert len(one_edge_graph.edges) == 1 and len(one_edge_graph.nodes) == 2, f'one_edge_graph has {len(one_edge_graph.edges)} edges and {len(one_edge_graph.nodes)} nodes'
        # torch.randperm(len(one_edge_graphs))
        # one_edge_graphs = [one_edge_graphs[i] for i in torch.randperm(len(one_edge_graphs))]
        return one_edge_graphs

    def __len__(self):
        return len(self.one_edge_graphs)

    def __getitem__(self, idx):
        return self.one_edge_graphs[idx]
    
    @staticmethod
    def collate_fn(batch):
        # batch is a list of dataset elements
        graphs = [item for item in batch]  # assuming your DiGraph is the first element in your dataset
        return graphs    

class TwoEdgeDataset(Dataset):
    def __init__(self, testonly=False):
        '''
        Dataset of two edge graphs, with two edges having one vertex in common in the same graph per sample
        '''
        super().__init__()
        self.graphs = load_filtered_graphs(testonly=testonly)
        print('Generating two edge graphs...')
        self.two_edge_graphs = self.generate_two_edge_graphs()
        print('Finished generating two edge graphs')

    def generate_two_edge_graphs(self):
        two_edge_graphs = []
        for graph in tqdm(self.graphs):
            paths = get_all_undir_paths(graph, length=2)
            for path in paths:
                two_edge_graph = copy_graph(graph, nodes_restrict=[path[0][0], path[0][1], path[1][0], path[1][1]], edges_restrict=[path[0], path[1]])
                two_edge_graphs.append(two_edge_graph)
                assert len(two_edge_graph.edges) == 2, f'graph {two_edge_graph} has {len(two_edge_graph.edges)} edges: {two_edge_graph.edges}'
        # torch.randperm(len(two_edge_graphs))
        # two_edge_graphs = [two_edge_graphs[i] for i in torch.randperm(len(two_edge_graphs))]
        return two_edge_graphs

    def __len__(self):
        return len(self.two_edge_graphs)

    def __getitem__(self, idx):
        return self.two_edge_graphs[idx]

    @staticmethod
    def collate_fn(batch):
        # batch is a list of dataset elements
        graphs = [item for item in batch]  # assuming your DiGraph is the first element in your dataset
        return graphs   

class FullGraphsDataset(Dataset):
    def __init__(self, testonly=False):
        '''
        Dataset of full graphs
        '''
        super().__init__()
        self.graphs = load_filtered_graphs(testonly=testonly)
        # torch.randperm(len(self.graphs))
        # self.graphs = [self.graphs[i] for i in torch.randperm(len(self.graphs))]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @staticmethod
    def collate_fn(batch):
        # batch is a list of dataset elements
        graphs = [item for item in batch]  # assuming your DiGraph is the first element in your dataset
        return graphs

class MSCOCOGraphsDataset(Dataset):
    def __init__(self, testonly=False):
        '''
        Dataset of full, raw graphs from the Visual Genome overlap with the MSCOCO evaluation set,
        annotated with ChatGPT.
        '''
        super().__init__()
        self.graphs = load_mscoco_graphs(testonly=testonly)
        # torch.randperm(len(self.graphs))
        # self.graphs = [self.graphs[i] for i in torch.randperm(len(self.graphs))]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @staticmethod
    def collate_fn(batch):
        # batch is a list of dataset elements
        graphs = [item for item in batch]  # assuming your DiGraph is the first element in your dataset
        return graphs

def get_one_edge_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = OneEdgeDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=OneEdgeDataset.collate_fn)

def get_two_edge_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = TwoEdgeDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=TwoEdgeDataset.collate_fn)

def get_full_graph_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = FullGraphsDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=FullGraphsDataset.collate_fn)

def get_mscoco_graph_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = MSCOCOGraphsDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=FullGraphsDataset.collate_fn)

def get_adversarial_relationship_dataset(version='v1'):
    download_adv_datasets()
    JONATHAN_DATASET_V1_PATH =  os.path.join(os.path.dirname(__file__), 'data', 'ra_selections_curated_adversarial.pt')
    JONATHAN_DATASET_V2_PATH =  os.path.join(os.path.dirname(__file__), 'data', 'ra_selections_curated_adversarial2.pt')
    if version == 'v1':
        curated_adversarialt = torch.load(JONATHAN_DATASET_V1_PATH) # a dict with image_id as key and a graph and the adversarial perturbations as value
    elif version == 'v2':
        curated_adversarialt = torch.load(JONATHAN_DATASET_V2_PATH)
    else:
        raise ValueError(f"version {version} not recognized")
    # the format of the dict is {image_id: [(original_graph,graph_edge,adv_predicate), ...]}
    # we return a list of tuples (graphs, adv_graph, adv_edge, adv_predicate) for each image for each adversarial perturbation
    dataset = []
    for original_graph, graph_edge, adv_predicate in curated_adversarialt:
        original_graph = copy_graph(original_graph, nodes_restrict=[graph_edge[0], graph_edge[1]], edges_restrict=[graph_edge])
        for node in original_graph.nodes:
            original_graph.nodes[node]['attributes'] = []
        adv_graph = copy_graph(original_graph)
        adv_graph.edges[graph_edge]['predicate'] = adv_predicate
        dataset.append({
            'original_graph': copy_graph(original_graph), # we copy to get the right captions
            'adv_graph': copy_graph(adv_graph)
        })
    return dataset

def get_adversarial_attribute_dataset(version='v1'):
    assert version == 'v1', f"version {version} not recognized"
    download_adv_datasets()
    ROBERT_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'realistic_adversarial_attributes_gt_accepted_pruned.json')
    with open(ROBERT_DATASET_PATH, 'r') as f:
        data = json.load(f)
    dataset = []
    for sample in data:
        graph = nx.DiGraph()
        graph.image_id = sample['image_id']
        obj1_name = sample['objects'][0]['names'][0]
        obj1_id = sample['objects'][0]['object_id']
        obj2_name = sample['objects'][1]['names'][0]
        obj2_id = sample['objects'][1]['object_id']
        obj1_attrs = sample['objects'][0]['attributes']
        obj2_attrs = sample['objects'][1]['attributes']
        assert obj1_name in FILTERED_OBJECTS, f"obj1_attrs: {obj1_attrs}"
        assert obj2_name in FILTERED_OBJECTS, f"obj2_attrs: {obj2_attrs}"
        for attr in obj1_attrs:
            assert attr in FILTERED_ATTRIBUTES, f"obj1_attrs: {obj1_attrs}"
        for attr in obj2_attrs:
            assert attr in FILTERED_ATTRIBUTES, f"obj2_attrs: {obj2_attrs}"
        # add nodes with attributes and labels
        graph.labels = {}
        graph.add_node(obj1_id, attributes=obj1_attrs, name=obj1_name)
        graph.labels[obj1_id] = obj1_name
        graph.add_node(obj2_id, attributes=obj2_attrs, name=obj2_name)
        #### apparently this is needed for Robert's code to work. But as and is not a valid filtered predicate, we don't do that
        # graph.add_edge(obj1_id, obj2_id, predicate='and')
        # graph.add_edge(obj2_id, obj1_id, predicate='and')
        ####
        graph.labels[obj2_id] = obj2_name
        graph_adv = copy_graph(graph)
        [n1,n2] = list(graph_adv.nodes)[0:2]
        graph_adv.nodes[n1]['attributes'], graph_adv.nodes[n2]['attributes'] = graph_adv.nodes[n2]['attributes'], graph_adv.nodes[n1]['attributes']
        dataset.append({
            'original_graph': copy_graph(graph), # we copy to get the right captions
            'adv_graph': copy_graph(graph_adv),
        })
    return dataset



def get_adv_prompt_list(type, version='v1'):
    assert type in ['relationships', 'attributes'], f"type {type} not recognized, should be 'relationships' or 'attributes'"
    dataset = get_adversarial_relationship_dataset(version=version) if type == 'relationships' else get_adversarial_attribute_dataset(version=version)
    prompt_list_original = []
    prompt_list_adv = []
    for sample in dataset:
        if type == 'attributes':
            n1,n2 = list(sample['original_graph'].nodes)[0:2]
            # add edge
            sample['original_graph'].add_edge(n1,n2, predicate='and')
            sample['adv_graph'].add_edge(n1,n2, predicate='and')
            sample['original_graph'] = copy_graph(sample['original_graph'])
            sample['adv_graph'] = copy_graph(sample['adv_graph'])
        prompt_list_original.append(sample['original_graph'].caption)
        prompt_list_adv.append(sample['adv_graph'].caption)
    return prompt_list_original, prompt_list_adv
