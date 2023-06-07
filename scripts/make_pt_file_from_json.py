from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import io
from PIL import Image
import json
import os
import torch
# you need to download the scene graph data from visual genome, it's not included in dario's folder (and i don't have write access there)
with open('datasets/visual_genome/raw/scene_graphs.json', 'r') as f:
    scene_graphs_dict = json.load(f)
    
with open('datasets/graph_captions/captions_2000_filtered.json', 'r') as f:
    captions_dict = json.load(f)
    captions_dict = {d['image_id']: d['captions']['short'] for d in captions_dict}
    
def build_graph(g_dict, text):
        G = nx.DiGraph()
        G.caption = text
        G.image_id=g_dict['image_id']
        path = 'datasets/visual_genome/raw/VG_100K/'+str(G.image_id)+'.jpg'
        if not os.path.exists(path):
            path = 'datasets/visual_genome/raw/VG_100K_2/'+str(G.image_id)+'.jpg'
        with open(path, 'rb') as f:
            image_bytes = f.read()
            s = Image.open(io.BytesIO(image_bytes)).size
            G.image_w = s[0]
            G.image_h = s[1]
        G.labels = {}
        for obj in g_dict['objects']:
            G.add_node(obj['object_id'], w=obj['w'], h=obj['h'], x=obj['x'], y=obj['y'], attributes=obj.get('attributes',[]), name=obj['names'][0])
            G.labels[obj['object_id']] = obj['names'][0]
        for rel in g_dict['relationships']:
            G.add_edge(rel['subject_id'], rel['object_id'], synsets=rel['synsets'] ,relationship_id=rel['relationship_id'], predicate=rel['predicate'])
        return G
graphs = [] 
for g_dict in tqdm(scene_graphs_dict):
    if g_dict['image_id'] not in captions_dict:
        continue
    graphs.append(build_graph(g_dict, captions_dict[g_dict['image_id']]))

# convert all object labels, attributes and predicates to lower case and remove trailing spaces
for g in tqdm(graphs):
    for n in g.nodes:
        g.nodes[n]['name'] = g.nodes[n]['name'].lower().strip()
        g.nodes[n]['attributes'] = [a.lower().strip() for a in g.nodes[n]['attributes']]
    for e in g.edges:
        g.edges[e]['predicate'] = g.edges[e]['predicate'].lower().strip()

torch.save(graphs, "mscoco_graphs.pt")
torch.save(graphs[:10], "mscoco_graphs_test_small.pt")