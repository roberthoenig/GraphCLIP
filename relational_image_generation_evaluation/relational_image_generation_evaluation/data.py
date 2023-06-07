from .download_weights import download_filtered_graphs
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

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

def get_one_edge_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = OneEdgeDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=OneEdgeDataset.collate_fn)

def get_two_edge_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = TwoEdgeDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=TwoEdgeDataset.collate_fn)

def get_full_graph_dataloader(batch_size=1, shuffle=True, testonly=False):
    dataset = FullGraphsDataset(testonly=testonly)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=FullGraphsDataset.collate_fn)