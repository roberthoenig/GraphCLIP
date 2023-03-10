import torch
from torch_geometric.data import InMemoryDataset, download_url
from utils.dataset_utils import unzip_file
import os.path as osp
import json
from tqdm import tqdm
import logging
import open_clip
from PIL import Image
from torch_geometric.data import Data
from pathlib import Path


def dict_to_pyg_graph(d, img_enc, txt_enc, image_id_to_path):
    # y: [1, num_img_features]
    # TODO: normalize?
    y = img_enc([image_id_to_path[d['image_id']]])
    # x: [num_nodes, num_txt_features]
    id_to_idx = {}
    # TODO: deal with multiple object names?
    # TODO: deal with object attributes?
    # TODO: normalize?
    x = txt_enc([obj['names'][0] for obj in d['objects']])
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.long)
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
    # edge_attr: [num_edges, num_txt_features]
    if len(d['relationships']) == 0:
        edge_attr = torch.zeros(0, 1024)
    else:
        rel_txts = []
        for rel in d['relationships']:
            subj_txt = d['objects'][id_to_idx[rel['subject_id']]]['names'][0]
            obj_txt = d['objects'][id_to_idx[rel['object_id']]]['names'][0]
            rel_txt = rel['predicate']
            compound_txt = " ".join([subj_txt, rel_txt, obj_txt])
            rel_txts.append(compound_txt)
        edge_attr = txt_enc(rel_txts)
    
    data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)
    return data

class VisualGenome(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, enc_cfg=None, n_samples="all"):
        self.enc_cfg = enc_cfg
        self.n_samples = n_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['scene_graphs.json.zip', 'images.zip', 'images2.zip']

    @property
    def processed_file_names(self):
        return [f"data_{self.n_samples}_{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        scene_graphs_url = "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"
        download_url(scene_graphs_url, self.raw_dir)
        unzip_file(self.raw_paths[0], self.raw_dir)
        
        images_1_url = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
        download_url(images_1_url, self.raw_dir)
        unzip_file(self.raw_paths[1], self.raw_dir)
        
        images_2_url = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
        download_url(images_2_url, self.raw_dir)
        unzip_file(self.raw_paths[2], self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        logging.info("Loading scene graph JSON file...")
        with open(osp.join(self.raw_dir, "scene_graphs.json"), 'r') as f:
            scene_graphs_dict = json.load(f)
        if not self.n_samples == "all":
            scene_graphs_dict = scene_graphs_dict[:self.n_samples]
        logging.info("Processing scene graphs into PyG graphs...")
        
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=self.enc_cfg["model_name"], pretrained=self.enc_cfg["pretrained"], device=self.enc_cfg["device"])
        tokenizer = open_clip.get_tokenizer(model_name=self.enc_cfg["model_name"])
        def img_enc(img_paths):
            with torch.no_grad():
                return model.encode_image(torch.stack([preprocess(Image.open(p)) for p in img_paths]).to(self.enc_cfg["device"])).cpu()
        def txt_enc(txts):
            with torch.no_grad():
                return model.encode_text(tokenizer(txts).to(self.enc_cfg["device"])).cpu()
        
        image_id_to_path = dict()
        for dir in [Path(self.raw_dir)/"VG_100K", Path(self.raw_dir)/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                image_id_to_path[img_id] = str(path)
            
        data_list = [dict_to_pyg_graph(d, img_enc, txt_enc, image_id_to_path) for d in tqdm(scene_graphs_dict)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.info("Collating PyG graphs...")
        data, slices = self.collate(data_list)

        logging.info("Saving PyG graphs...")
        torch.save((data, slices), self.processed_paths[0])