import json
import os
import pandas as pd

class MSCOCO:
    def __init__(self, n_samples, image_path, ann_path, id_path,  caption_path=None):
        with open(ann_path,  'r') as f:
            ann = json.load(f)
        with open(id_path, 'r') as f:
            ids = json.load(f)
        imageid2filename = {item['id']:item['file_name'] for item in ann['images']}
        n_samples = len(ann['images']) if n_samples == "all" else n_samples
        assert n_samples <= len(ann['images'])
        df = pd.DataFrame.from_dict(ann['annotations'])
        images   = []
        captions = []
        if caption_path is None:
            for image_id, group in df.groupby('image_id'):
                if image_id not in ids:
                    continue
                images.append(imageid2filename[image_id])
                captions.append(list(group['caption'])[:5])
                if len(images) >= n_samples:
                    break
        else:
            with open(caption_path, 'r') as f:
                captions_j = json.load(f)
            # WARNING: There are several samples with identical coco_id's. The following
            # dictionary comprehension only keeps the last of them.
            print(captions_j[0])
            captions_d = {c['coco_id']: c['captions']['short'] if not c.get('info', None)=='Empty Graph!' else "d" for c in captions_j}
            captions = []
            for image_id, group in df.groupby('image_id'):
                if image_id not in ids:
                    continue
                maybe_caption = captions_d.get(image_id, None)
                if maybe_caption is None:
                    continue
                images.append(imageid2filename[image_id])
                captions.append([maybe_caption])
                if len(images) >= n_samples:
                    break
        self.img_paths = [os.path.join(image_path, image) for image in images]
        self.captions = captions
