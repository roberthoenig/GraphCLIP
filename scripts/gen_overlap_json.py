# Computes the overlap between the MSCOCO 2017 validation set and Visual Genome.
import os.path as osp
import json
from pathlib import Path

BASE_DIR = "datasets/visual_genome/raw"
OUT_DIR = "datasets/mscoco"

print("Loading coco_val_ids")
with open(osp.join(BASE_DIR, "annotations", "instances_val2017.json"), 'r') as f:
    instances_val2017 = json.load(f)
coco_val_ids = {int(Path(o['file_name']).stem) for o in instances_val2017['images']}

print("Loading visual_genome_ids")
with open(osp.join(BASE_DIR, "image_data.json"), 'r') as f:
    image_data = json.load(f)
visual_genome_ids = {o['coco_id'] for o in image_data if o['coco_id'] is not None}

print("Computing overlap")
overlap = visual_genome_ids.intersection(coco_val_ids)
overlap = sorted(list(overlap))
print(f"{len(overlap)} overlaps.")
with open(osp.join(OUT_DIR, 'overlap.json'), 'w') as f:
    json.dump(overlap, f)