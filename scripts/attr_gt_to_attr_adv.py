import json
from copy import deepcopy

ATTR_GT_PATH = "datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json"
ATTR_ADV_PATH = "datasets/visual_genome/raw/realistic_adversarial_attributes_adv_accepted_pruned.json"

# Save all newly created graphs
with open(ATTR_GT_PATH, 'r') as f:
    gt_graphs = json.load(f)
    
# Swap attributes in graphs
adv_graphs = []
for graph in gt_graphs:
    adv_graph = deepcopy(graph)
    adv_graph['objects'][0]['attributes'], adv_graph['objects'][1]['attributes'] = \
        adv_graph['objects'][1]['attributes'], adv_graph['objects'][0]['attributes'] 
    adv_graphs.append(adv_graph)

# Save adversarial graphs
with open(ATTR_ADV_PATH, 'w') as f:
    json.dump(adv_graphs, f)
