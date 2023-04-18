import json

CAPTIONS_PATH = "datasets/graph_captions/captions_2000.json"
CAPTIONS_FILTERED_PATH = "datasets/graph_captions/captions_2000_filtered.json"
SCENE_GRAPHS_PATH = 'datasets/visual_genome/raw/scene_graphs.json'

with open(CAPTIONS_PATH, "r") as f:
    captions = json.load(f)

with open(SCENE_GRAPHS_PATH, 'r') as f:
    scene_graphs_dict = json.load(f)

# Filter
has_nodes = dict()
for d in scene_graphs_dict:
    has_node = len(d['objects']) > 0
    has_nodes[d['image_id']] = has_node
captions = [c for c in captions if has_nodes[c['image_id']]]

# Transform
fn = lambda d: d["n_tokens_short"]
for c in captions:
    if c.get('info', None) == 'Empty Graph!':
        print("SHOULD NOT HAPPEN")
        exit(1)
    c['captions'].sort(key=fn)
    c['captions'] = [d for d in c['captions'] if d["n_tokens_short"] <= 80][-1]

with open(CAPTIONS_FILTERED_PATH, "w") as f:
    captions = json.dump(captions, f)