import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import json
import random
from PIL import Image


LOCAL_DATA_PATH = '/local/home/jthomm/GraphCLIP/datasets/visual_genome/'

def copy_graph(g):
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
    return g_copy

def plot_graph(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    max_y = max([y for x,y in pos.values()])
    n_nodes_top = len([n for n in g.nodes if pos[n][1] == max_y])
    longest_label = max([len(g.labels[n]) for n in g.nodes])
    plt.figure(figsize=(max(n_nodes_top*longest_label/10,15),5))
    nx.draw(g,pos=pos,labels=g.labels, with_labels=True, node_size=10, node_color="lightgray", font_size=8)
    nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=nx.get_edge_attributes(g,'predicate'),font_size=8)
    plt.show()

def get_image(image_id):
    '''returns the image with the given id'''
    image_dir = "/local/home/stuff/visual-genome/VG/"
    image_path = image_dir+str(image_id)+'.jpg'
    image = Image.open(image_path)
    return image

def convert_adversarially(g, relationship_labels):
    '''returns a copy of the graph g with one edge replaced by a random edge from the relationship labels'''
    # convert to adversarial graph by looking for one edge that is in the relationship labels and replacing it with a random one from the relationship labels
    # we remove trailing spaces and convert to lower case
    g = copy_graph(g)
    indexes = [i for i, e in enumerate(g.edges) if g.edges[e]['predicate'].strip().lower() in relationship_labels]
    subject = None
    object = None
    relationship = None
    if len(indexes) > 0:
        idx = random.choice(indexes)
        e = list(g.edges)[idx]
        subject = g.nodes[e[0]]['name']
        object = g.nodes[e[1]]['name']
        new_predicate = random.choice(relationship_labels)
        relationship = (g.edges[e]['predicate'],new_predicate)
        g.edges[e]['predicate'] = new_predicate
    return g, subject, object, relationship


def get_filtered_graphs(test_small=False):
    '''returns a list of netwrorkx graphs, each graph is a scene graph from visual genome'''
    # check if filtered graphs are already saved, if not create them
    try:
        if test_small:
            filtered_graphs = torch.load(LOCAL_DATA_PATH+'processed/filtered_graphs_test_small.pt')
        else:
            filtered_graphs = torch.load(LOCAL_DATA_PATH+'processed/filtered_graphs.pt')
        print('Filtered graphs loaded from file')
    except:
        print('Filtered graphs not found, creating them')
        create_filtered_graphs()
        filtered_graphs = torch.load(LOCAL_DATA_PATH+'processed/filtered_graphs.pt')
    return filtered_graphs


def get_filtered_relationships():
    '''returns a list of strings, each string is a filtered relationship label from visual genome'''
    # check if filtered relationships are already saved, if not create them
    try:
        relationship_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_relationship_labels.pt')
        print('Filtered relationships loaded from file')
    except:
        print('Filtered relationships not found, creating them')
        create_filtered_graphs()
        relationship_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_relationship_labels.pt')
    return relationship_labels


def get_filtered_objects():
    '''returns a list of strings, each string is a filtered object label from visual genome'''
    # check if filtered objects are already saved, if not create them
    try:
        object_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_object_labels.pt')
        print('Filtered objects loaded from file')
    except:
        print('Filtered objects not found, creating them')
        create_filtered_graphs()
        object_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_object_labels.pt')
    return object_labels

def get_filtered_attributes():
    '''returns a list of strings, each string is a filtered attribute label from visual genome'''
    # check if filtered attributes are already saved, if not create them
    try:
        attribute_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_attribute_labels.pt')
        print('Filtered attributes loaded from file')
    except:
        print('Filtered attributes not found, creating them')
        create_filtered_graphs()
        attribute_labels = torch.load(LOCAL_DATA_PATH+'processed/filtered_attribute_labels.pt')
    return attribute_labels


def create_filtered_graphs():
    raise NotImplementedError('This function is out of date.')
    # you need to download the scene graph data from visual genome, it's not included in dario's folder (and i don't have write access there)
    with open(LOCAL_DATA_PATH+'raw/scene_graphs.json', 'r') as f:
        scene_graphs_dict = json.load(f)

    def build_graph(g_dict):
            G = nx.DiGraph()
            G.image_id=g_dict['image_id']
            G.labels = {}
            for obj in g_dict['objects']:
                G.add_node(obj['object_id'], w=obj['w'], h=obj['h'], x=obj['x'], y=obj['y'], attributes=obj.get('attributes',[]), name=obj['names'][0])
                G.labels[obj['object_id']] = obj['names'][0]
            for rel in g_dict['relationships']:
                G.add_edge(rel['subject_id'], rel['object_id'], synsets=rel['synsets'] ,relationship_id=rel['relationship_id'], predicate=rel['predicate'])
            return G
    graphs = [] 
    for g_dict in tqdm(scene_graphs_dict):
        graphs.append(build_graph(g_dict))
    # convert all object labels, attributes and predicates to lower case and remove trailing spaces
    for g in tqdm(graphs):
        for n in g.nodes:
            g.nodes[n]['name'] = g.nodes[n]['name'].lower().strip()
            g.nodes[n]['attributes'] = [a.lower().strip() for a in g.nodes[n]['attributes']]
        for e in g.edges:
            g.edges[e]['predicate'] = g.edges[e]['predicate'].lower().strip()
    
    # extract all object labels and all relationship labels and all attribute labels from the graphs
    object_labels = {}
    relationship_labels = {}
    attribute_labels = {}
    for g in graphs:
        for obj_label in g.labels.values():
            # # remove trailing spaces
            # obj_label = obj_label.strip().lower()
            object_labels[obj_label] = object_labels.get(obj_label, 0) + 1
        for rel_label in [g.edges[e]['predicate'] for e in g.edges]:
            # rel_label = rel_label.strip().lower()
            relationship_labels[rel_label] = relationship_labels.get(rel_label, 0) + 1
        for attr_label in [g.nodes[n]['attributes'] for n in g.nodes]:
            for a in attr_label:
                # a = a.strip().lower()
                attribute_labels[a] = attribute_labels.get(a, 0) + 1
    print(f'number of graphs: {len(graphs)}')
    print(f'number of object labels: {len(object_labels)}')
    print(f'number of relationship labels: {len(relationship_labels)}')
    print(f'number of attribute labels: {len(attribute_labels)}')
    # sort the labels for both the objects and relationships and extract the 100 most frequent ones in the graphs
    object_labels_occurrences = sorted(object_labels.items(), key=lambda x: x[1], reverse=True)[:200]
    relationship_labels_occurrences = sorted(relationship_labels.items(), key=lambda x: x[1], reverse=True)[:100]
    attribute_labels_occurrences = sorted(attribute_labels.items(), key=lambda x: x[1], reverse=True)[:100]
    # extract the labels from the tuples
    object_labels = [l[0] for l in object_labels_occurrences]
    relationship_labels = [l[0] for l in relationship_labels_occurrences]
    attribute_labels = [l[0] for l in attribute_labels_occurrences]
    def build_filtered_graph(g, object_labels, relationship_labels, attribute_labels):
            G = nx.DiGraph()
            G.image_id=g.image_id
            G.labels = {}
            for n in g.nodes:
                if g.labels[n] in object_labels:
                    G.add_node(n, w=g.nodes[n]['w'], h=g.nodes[n]['h'], x=g.nodes[n]['x'], y=g.nodes[n]['y'], attributes=g.nodes[n]['attributes'].copy(), name=g.labels[n])
                    G.labels[n] = g.labels[n]
            for e in g.edges:
                if g.edges[e]['predicate'] in relationship_labels and e[0] in G.nodes and e[1] in G.nodes:
                    G.add_edge(e[0], e[1], synsets=g.edges[e]['synsets'].copy() ,relationship_id=g.edges[e]['relationship_id'], predicate=g.edges[e]['predicate'])
            return G

    # filter the graphs relationships and objects and attributes to only keep the 100/200 most frequent ones. Remove graphs which have no objects or relationships left after filtering
    filtered_graphs = []
    for g in tqdm(graphs):
        g_filtered = build_filtered_graph(g, object_labels, relationship_labels, attribute_labels)
        if len(g_filtered.nodes) > 0 and len(g_filtered.edges) > 0:
            filtered_graphs.append(g_filtered)
    # print the graphs stats: number of graphs, number of objects, number of relationships, number of attributes
    print(f'number of graphs: {len(filtered_graphs)}')
    print(f'number of objects: {sum([len(g.nodes) for g in filtered_graphs])}')
    print(f'number of relationships: {sum([len(g.edges) for g in filtered_graphs])}')
    print(f'number of attributes: {sum([len(g.nodes[n]["attributes"]) for g in filtered_graphs for n in g.nodes])}')
    torch.save(filtered_graphs, LOCAL_DATA_PATH+'processed/filtered_graphs.pt')
    torch.save(object_labels, LOCAL_DATA_PATH+'processed/filtered_object_labels.pt')
    torch.save(relationship_labels, LOCAL_DATA_PATH+'processed/filtered_relationship_labels.pt')
    torch.save(attribute_labels, LOCAL_DATA_PATH+'processed/filtered_attribute_labels.pt')
