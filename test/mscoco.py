from datasets.mscoco import MSCOCO

image_path = "datasets/mscoco/annotations_trainval2017/val2017"
ann_path = "datasets/mscoco/annotations_trainval2017/annotations/captions_val2017.json"
dataset = MSCOCO(n_samples="all", image_path=image_path, ann_path=ann_path)