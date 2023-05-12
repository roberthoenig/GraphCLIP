# 1. Load graphs from graph file "datasets/visual_genome/raw/scene_graphs.json"
# 2. Load image description file "datasets/visual_genome/raw/image_data.json"
# 3. Filter graphs to all MSCOCO graphs
# 4. Loop through graphs
# 5. Filter to all entities with bounding box occupying at least 5% of the image.
# 6. Filter to all entities whose names appear only once in the graph.
# 7. Loop over all pairs of entities.
# 8. Filter to all pairs of entities with an edge between them.
# 9. For each pair of entities (e1, e2), filter their sets of attributes so that they do not overlap.
# 10. Filter to all pairs of attributes (a1, a2) such that G contains at least one entity with the same name as e1 that has attribute a2 and at least one entity with the same name as e2 that has attribute a1.
# 11. Create a new graph that contains only e1 and e2 with  attributes a1 and a2 swapped. Include the edge between e1 and e2.
# 12. Save all newly created graphs to file "datasets/visual_genome/raw/realistic_adversarial_attributes.json"

import json
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# SCENE_GRAPHS_PATH = "test/create_adversarial_attributes_dataset_test/scene_graph_1.json"
# IMAGE_DATA_PATH = "test/create_adversarial_attributes_dataset_test/image_data_1.json"
SCENE_GRAPHS_PATH = "datasets/visual_genome/raw/scene_graphs.json"
IMAGE_DATA_PATH = "datasets/visual_genome/raw/image_data.json"

# Load graphs and image descriptions
print(f"Loading {SCENE_GRAPHS_PATH}")
with open(SCENE_GRAPHS_PATH) as f:
    graphs = json.load(f)

print(f"Loading {IMAGE_DATA_PATH}")
with open(IMAGE_DATA_PATH) as f:
    images = json.load(f)

# Mapping image id to image for faster lookup
print("Mapping image ids to images")
image_dict = {image['image_id']: image for image in images}


# Mapping object names to the possible attributes they can have
print("Mapping object names to attributes")
name_to_attrs = defaultdict(set)
for graph in graphs:
    for obj in graph['objects']:
        for name in obj['names']:
            for attr in obj.get('attributes', []):
                name_to_attrs[name].add(attr)
            

# Filter graphs to all MSCOCO graphs (Assuming MSCOCO graphs are those with a valid coco_id)
print("Filtering graphs to MSCOCO")
with open("datasets/visual_genome/raw/annotations/instances_val2017.json", 'r') as f:
    mscoco_val_dict = json.load(f)
    coco_val_ids = set([int(Path(o['file_name']).stem) for o in mscoco_val_dict['images']])
graphs = [graph for graph in graphs if image_dict[graph['image_id']]['coco_id'] in coco_val_ids]

new_graphs = []

# Loop through graphs
print("Looping through graphs")
for graph in tqdm(graphs):
    appended = 0
    image = image_dict[graph['image_id']]
    image_area = image['width'] * image['height']
    
    # Filter entities
    entities = [obj for obj in graph['objects'] 
                # bounding box must occupy at least 5% of the image
                if obj['h'] * obj['w'] / image_area >= 0.05 
                # object names must be unique across the graph
                and all(len(set(obj['names']).intersection(obj2['names']))==0 or obj2["object_id"]==obj["object_id"] for obj2 in graph['objects'])]
    # Loop over all pairs of entities
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            e1, e2 = entities[i], entities[j]
            
            # Compute edges between e1 and e2
            e1_e2_relationships = [rel for rel in graph['relationships'] if
                rel['object_id'] in {e1['object_id'], e2['object_id']} 
                and rel['subject_id'] in {e1['object_id'], e2['object_id']} ]
            # Proceed only if there exists at least one edge between e1 and e2
            if len(e1_e2_relationships) > 0:
                # Filter attributes to not overlap
                e1_filtered_attrs = [attr for attr in e1.get('attributes', []) if
                    # e2 must not have attribute attr
                    attr not in e2.get('attributes', []) and
                    # at least one object named like e2 must have attribute attr 
                    any(attr in name_to_attrs[name] for name in e2['names'])]
                e2_filtered_attrs = [attr for attr in e2.get('attributes', []) if
                    # e1 must not have attribute attr
                    attr not in e1.get('attributes', []) and
                    # at least one object named like e1 must have attribute attr 
                    any(attr in name_to_attrs[name] for name in e1['names'])
                    ]
                # Loop over all pairs of attributes
                for a1 in e1_filtered_attrs:
                    for a2 in e2_filtered_attrs:
                        assert a1 != a2
                        # Create a new graph that contains only e1 and e2 with attributes a1 and a2
                        new_e1 = deepcopy(e1)
                        new_e1['attributes'] = [a1]
                        new_e2 = deepcopy(e2)
                        new_e2['attributes'] = [a2]
                        new_graph = {
                            'image_id': graph['image_id'],
                            'relationships': e1_e2_relationships,
                            'objects': [new_e1, new_e2]
                        }
                        new_graphs.append(new_graph)
                        appended += 1
    print(f"{graph['image_id']}: {appended} samples extracted.")

# Save all newly created graphs
with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt.json", 'w') as f:
    json.dump(new_graphs, f)
    
# Swap attributes in graphs
adv_graphs = []
for graph in new_graphs:
    adv_graph = deepcopy(graph)
    adv_graph['objects'][0]['attributes'], adv_graph['objects'][1]['attributes'] = \
        adv_graph['objects'][1]['attributes'], adv_graph['objects'][0]['attributes'] 
    adv_graphs.append(adv_graph)

# Save adversarial graphs
with open("datasets/visual_genome/raw/realistic_adversarial_attributes_adv.json", 'w') as f:
    json.dump(adv_graphs, f)
