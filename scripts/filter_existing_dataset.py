import json
from pathlib import Path

most_common_objs = ['man', 'person', 'window', 'tree', 'building', 'shirt', 'wall', 'woman', 'sign', 'sky', 'ground', 'grass', 'table', 'pole', 'head', 'light', 'water', 'car', 'hand', 'hair', 'people', 'leg', 'trees', 'clouds', 'ear', 'plate', 'leaves', 'fence', 'door', 'pants', 'eye', 'train', 'chair', 'floor', 'road', 'street', 'hat', 'snow', 'wheel', 'shadow', 'jacket', 'nose', 'boy', 'line', 'shoe', 'clock', 'sidewalk', 'boat', 'tail', 'cloud', 'handle', 'letter', 'girl', 'leaf', 'horse', 'bus', 'helmet', 'bird', 'giraffe', 'field', 'plane', 'flower', 'elephant', 'umbrella', 'dog', 'shorts', 'arm', 'zebra', 'face', 'windows', 'sheep', 'glass', 'bag', 'cow', 'bench', 'cat', 'food', 'bottle', 'rock', 'tile', 'kite', 'tire', 'post', 'number', 'stripe', 'surfboard', 'truck', 'logo', 'glasses', 'roof', 'skateboard', 'motorcycle', 'picture', 'flowers', 'bear', 'player', 'foot', 'bowl', 'mirror', 'background', 'pizza', 'bike', 'shoes', 'spot', 'tracks', 'pillow', 'shelf', 'cap', 'mouth', 'box', 'jeans', 'dirt', 'lights', 'legs', 'house', 'part', 'trunk', 'banana', 'top', 'plant', 'cup', 'counter', 'board', 'bed', 'wave', 'bush', 'ball', 'sink', 'button', 'lamp', 'beach', 'brick', 'flag', 'neck', 'sand', 'vase', 'writing', 'wing', 'paper', 'seat', 'lines', 'reflection', 'coat', 'child', 'toilet', 'laptop', 'airplane', 'letters', 'glove', 'vehicle', 'phone', 'book', 'branch', 'sunglasses', 'edge', 'cake', 'desk', 'rocks', 'frisbee', 'tie', 'tower', 'animal', 'hill', 'mountain', 'headlight', 'ceiling', 'cabinet', 'eyes', 'stripes', 'wheels', 'lady', 'ocean', 'racket', 'container', 'skier', 'keyboard', 'towel', 'frame', 'windshield', 'hands', 'back', 'track', 'bat', 'finger', 'pot', 'orange', 'fork', 'waves', 'design', 'feet', 'basket', 'fruit', 'broccoli', 'engine', 'guy', 'knife', 'couch', 'railing', 'collar', 'cars']

with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json", "r") as f:
    samples = json.load(f)
    
  
IMAGE_DATA_PATH = "datasets/visual_genome/raw/image_data.json"  
print(f"Loading {IMAGE_DATA_PATH}")
with open(IMAGE_DATA_PATH) as f:
    images = json.load(f)

# Mapping image id to image for faster lookup
print("Mapping image ids to images")
image_dict = {image['image_id']: image for image in images}
print("Filtering graphs to MSCOCO")
with open("datasets/visual_genome/raw/annotations/instances_val2017.json", 'r') as f:
    mscoco_val_dict = json.load(f)
    coco_val_ids = set([int(Path(o['file_name']).stem) for o in mscoco_val_dict['images']])
graphs = [graph for graph in samples if image_dict[graph['image_id']]['coco_id'] in coco_val_ids]
print("len(graphs)", len(graphs))
    
def has_common_names(sample):
    return all(any(name in most_common_objs for name in obj['names']) for obj in sample['objects'])
    
def store_only_first_common_name(sample):
    for obj in sample['objects']:
        for name in obj['names']:
            if name in most_common_objs:
                obj['names'] = [name]
                break
    return sample
    
new_samples = list(filter(has_common_names, samples))
new_samples = list(map(store_only_first_common_name, new_samples))

print("len(samples)", len(samples))
print("len(new_samples)", len(new_samples))

with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json", "w") as f:
    json.dump(new_samples, f)