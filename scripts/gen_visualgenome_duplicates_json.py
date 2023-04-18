# Computes the Visual Genome samples corresponding to each COCO sample.
# The mapping is mostly 1-1, but there are a few COCO samples with duplicate
# Visual Genome samples.
import json
from collections import defaultdict

OUT_PATH = "datasets/visual_genome/raw/visualgenome_duplicates.json"
IMAGE_DATA_PATH = 'datasets/visual_genome/raw/image_data.json'

with open(IMAGE_DATA_PATH, 'r') as f:
    image_data_dict = json.load(f)

vg_dupes = defaultdict(list)

for d in image_data_dict:
    coco_id = d['coco_id']
    image_id = d['image_id']
    if coco_id is not None:
        vg_dupes[coco_id].append(image_id)

for v in vg_dupes.values():
    v.sort()

with open(OUT_PATH, 'w') as f:
    json.dump(vg_dupes, f)