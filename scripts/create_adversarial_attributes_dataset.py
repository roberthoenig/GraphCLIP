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
from pprint import pprint

# SCENE_GRAPHS_PATH = "test/create_adversarial_attributes_dataset_test/scene_graph_1.json"
# IMAGE_DATA_PATH = "test/create_adversarial_attributes_dataset_test/image_data_1.json"
SCENE_GRAPHS_PATH = "datasets/visual_genome/raw/scene_graphs.json"
IMAGE_DATA_PATH = "datasets/visual_genome/raw/image_data.json"

# Flags
ONLY_REALISTIC_SWAPS = False
ONLY_COMMON_OBJECTS = False
ONLY_COMMON_ATTRIBUTES = False

most_common_attrs = ['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'wooden', 'gray', 'silver', 'metal', 'orange', 'grey', 'tall', 'long', 'dark', 'pink', 'clear', 'standing', 'round', 'tan', 'glass', 'here', 'wood', 'open', 'purple', 'big', 'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young', 'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy', 'colorful', 'one', 'beige', 'bare', 'wet', 'light', 'square', 'little', 'closed', 'stone', 'blonde', 'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted', 'thick', 'part', 'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular', 'looking', 'grassy', 'dry', 'light brown', 'cement', 'leafy', 'wearing', 'tiled', "man's", 'light blue', 'baseball', 'cooked', 'pictured', 'curved', 'decorative', 'dead', 'eating', 'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt', 'watching', 'colored', 'stuffed', 'circular']

most_common_objs = ['man', 'person', 'window', 'tree', 'building', 'shirt', 'wall', 'woman', 'sign', 'sky', 'ground', 'grass', 'table', 'pole', 'head', 'light', 'water', 'car', 'hand', 'hair', 'people', 'leg', 'trees', 'clouds', 'ear', 'plate', 'leaves', 'fence', 'door', 'pants', 'eye', 'train', 'chair', 'floor', 'road', 'street', 'hat', 'snow', 'wheel', 'shadow', 'jacket', 'nose', 'boy', 'line', 'shoe', 'clock', 'sidewalk', 'boat', 'tail', 'cloud', 'handle', 'letter', 'girl', 'leaf', 'horse', 'bus', 'helmet', 'bird', 'giraffe', 'field', 'plane', 'flower', 'elephant', 'umbrella', 'dog', 'shorts', 'arm', 'zebra', 'face', 'windows', 'sheep', 'glass', 'bag', 'cow', 'bench', 'cat', 'food', 'bottle', 'rock', 'tile', 'kite', 'tire', 'post', 'number', 'stripe', 'surfboard', 'truck', 'logo', 'glasses', 'roof', 'skateboard', 'motorcycle', 'picture', 'flowers', 'bear', 'player', 'foot', 'bowl', 'mirror', 'background', 'pizza', 'bike', 'shoes', 'spot', 'tracks', 'pillow', 'shelf', 'cap', 'mouth', 'box', 'jeans', 'dirt', 'lights', 'legs', 'house', 'part', 'trunk', 'banana', 'top', 'plant', 'cup', 'counter', 'board', 'bed', 'wave', 'bush', 'ball', 'sink', 'button', 'lamp', 'beach', 'brick', 'flag', 'neck', 'sand', 'vase', 'writing', 'wing', 'paper', 'seat', 'lines', 'reflection', 'coat', 'child', 'toilet', 'laptop', 'airplane', 'letters', 'glove', 'vehicle', 'phone', 'book', 'branch', 'sunglasses', 'edge', 'cake', 'desk', 'rocks', 'frisbee', 'tie', 'tower', 'animal', 'hill', 'mountain', 'headlight', 'ceiling', 'cabinet', 'eyes', 'stripes', 'wheels', 'lady', 'ocean', 'racket', 'container', 'skier', 'keyboard', 'towel', 'frame', 'windshield', 'hands', 'back', 'track', 'bat', 'finger', 'pot', 'orange', 'fork', 'waves', 'design', 'feet', 'basket', 'fruit', 'broccoli', 'engine', 'guy', 'knife', 'couch', 'railing', 'collar', 'cars']

def main(type):
    if type == "1":
        MIN_AREA = 0.05
        STRICT_UNIQUENESS = True
        USE_AND_RELATIONSHIP = False
        ONLY_REALISTIC_SWAPS = False
    elif type == "2":
        MIN_AREA = 0.00
        STRICT_UNIQUENESS = False
        USE_AND_RELATIONSHIP = True
        ONLY_REALISTIC_SWAPS = False
    elif type == "3":
        MIN_AREA = 0.00
        STRICT_UNIQUENESS = False
        USE_AND_RELATIONSHIP = True
        ONLY_REALISTIC_SWAPS = True
        
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
    # print("Filtering graphs to MSCOCO")
    # with open("datasets/visual_genome/raw/annotations/instances_val2017.json", 'r') as f:
    #     mscoco_val_dict = json.load(f)
    #     coco_val_ids = set([int(Path(o['file_name']).stem) for o in mscoco_val_dict['images']])
    # graphs = [graph for graph in graphs if image_dict[graph['image_id']]['coco_id'] in coco_val_ids]

    new_graphs = []

    # Loop through graphs
    print("Looping through graphs")
    for graph in tqdm(graphs):
        image = image_dict[graph['image_id']]
        image_area = image['width'] * image['height']
        
        # Filter entities
        entities = [obj for obj in graph['objects'] 
                    # bounding box must occupy at least 5% of the image
                    if obj['h'] * obj['w'] / image_area >= MIN_AREA
                    # object names must be unique across the graph
                    and ((not STRICT_UNIQUENESS) or (all(len(set(obj['names']).intersection(obj2['names']))==0 or obj2["object_id"]==obj["object_id"] for obj2 in graph['objects'])))
                    and ((not ONLY_COMMON_OBJECTS) or any(name in most_common_objs for name in obj['names']))]
        # Loop over all pairs of entities
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if e1['names'][0] == e2['names'][0]:
                    continue
                
                # Compute edges between e1 and e2
                if USE_AND_RELATIONSHIP:
                    e1_e2_relationships = [
                        {
                            "predicate": "and",
                            "object_id": e1['object_id'],
                            "subject_id": e2['object_id'],
                        },
                        {
                            "predicate": "and",
                            "object_id": e2['object_id'],
                            "subject_id": e1['object_id'],
                        },
                    ]
                else: 
                    e1_e2_relationships = [rel for rel in graph['relationships'] if
                        rel['object_id'] in {e1['object_id'], e2['object_id']} 
                        and rel['subject_id'] in {e1['object_id'], e2['object_id']} ]
                # Filter attributes to not overlap
                e1_filtered_attrs = [attr for attr in e1.get('attributes', []) if
                    # e2 must not have attribute attr
                    attr not in e2.get('attributes', []) and
                    # at least one object named like e2 must have attribute attr 
                    ((not ONLY_REALISTIC_SWAPS) or any(attr in name_to_attrs[name] for name in e2['names'])) and
                    ((not ONLY_COMMON_ATTRIBUTES) or attr in most_common_attrs)]
                e2_filtered_attrs = [attr for attr in e2.get('attributes', []) if
                    # e1 must not have attribute attr
                    attr not in e1.get('attributes', []) and
                    # at least one object named like e1 must have attribute attr 
                    ((not ONLY_REALISTIC_SWAPS) or any(attr in name_to_attrs[name]) for name in e1['names']) and
                    ((not ONLY_COMMON_ATTRIBUTES) or attr in most_common_attrs)]
                # Loop over all pairs of attributes
                for a1 in e1_filtered_attrs:
                    for a2 in e2_filtered_attrs:
                        assert a1 != a2
                        # Create a new graph that contains only e1 and e2 with attributes a1 and a2
                        new_graph = {
                            'image_id': graph['image_id'],
                            'relationships': e1_e2_relationships,
                            'objects': [
                                {
                                    "object_id": e1['object_id'],
                                    "names": e1['names'][:1],
                                    "attributes": e1['attributes'][:1],
                                },
                                {
                                    "object_id": e2['object_id'],
                                    "names": e2['names'][:1],
                                    "attributes": e2['attributes'][:1],
                                }
                            ]
                        }
                        new_graphs.append(new_graph)
        # print(f"{graph['image_id']}: {appended} samples extracted.")

    print("len(new_graphs)", len(new_graphs))
    print("Saving new graphs...")
    # Save all newly created graphs
    with open(f"datasets/visual_genome/raw/realistic_adversarial_attributes_gt_{type}.json", 'w') as f:
        json.dump(new_graphs, f)
        
        
    # # Swap attributes in graphs
    # adv_graphs = []
    # for graph in new_graphs:
    #     adv_graph = deepcopy(graph)
    #     adv_graph['objects'][0]['attributes'], adv_graph['objects'][1]['attributes'] = \
    #         adv_graph['objects'][1]['attributes'], adv_graph['objects'][0]['attributes'] 
    #     adv_graphs.append(adv_graph)

    # # Save adversarial graphs
    # with open("datasets/visual_genome/raw/realistic_adversarial_attributes_adv.json", 'w') as f:
    #     json.dump(adv_graphs, f)

if __name__ == '__main__':
    main(type="3")