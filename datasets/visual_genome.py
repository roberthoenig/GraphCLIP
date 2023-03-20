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

# Embeds text with CLIP
def dict_to_pyg_graph(d, img_enc, txt_enc, image_id_to_path, emb_dim, metadata):
    # y: [1, num_img_features]
    # TODO: normalize?
    y = img_enc(image_id_to_path[d['image_id']])
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
        edge_attr = torch.zeros(0, emb_dim)
    else:
        rel_txts = []
        for rel in d['relationships']:
            subj_txt = d['objects'][id_to_idx[rel['subject_id']]]['names'][0]
            obj_txt = d['objects'][id_to_idx[rel['object_id']]]['names'][0]
            rel_txt = rel['predicate']
            compound_txt = " ".join([subj_txt, rel_txt, obj_txt])
            rel_txts.append(compound_txt)
        edge_attr = txt_enc(rel_txts)
    
    num = metadata['coco_id'] if metadata['coco_id'] is not None else -1
    data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y, coco_id=torch.tensor([num], dtype=torch.long))
    return data

class VisualGenome(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, enc_cfg=None, n_samples="all"):
        self.enc_cfg = enc_cfg
        self.n_samples = n_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['scene_graphs.json.zip', 'images.zip', 'images2.zip', 'image_data.json.zip']

    @property
    def processed_file_names(self):
        return [f"data_{self.n_samples}_{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}_use_clip_latents={self.enc_cfg['use_clip_latents']}_coco_annotated.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        def download_and_unzip_if_not_exist(url, idx):
            path = osp.join(self.raw_dir, self.raw_file_names[idx])
            if not osp.isfile(path):
                download_url(url, self.raw_dir)
                unzip_file(self.raw_paths[idx], self.raw_dir)
            else:
                print(f"{path} already exists, skipping download.")
        download_and_unzip_if_not_exist("http://visualgenome.org/static/data/dataset/scene_graphs.json.zip", 0)
        download_and_unzip_if_not_exist("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", 1)
        download_and_unzip_if_not_exist("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", 2)
        download_and_unzip_if_not_exist("http://visualgenome.org/static/data/dataset/image_data.json.zip", 3)

    def process(self):
        # Read data into huge `Data` list.
        logging.info("Loading scene graph JSON file...")
        with open(osp.join(self.raw_dir, "scene_graphs.json"), 'r') as f:
            scene_graphs_dict = json.load(f)
        logging.info("Loading image data JSON file...")
        with open(osp.join(self.raw_dir, "image_data.json"), 'r') as f:
            image_data_dict = json.load(f)
        if not self.n_samples == "all":
            scene_graphs_dict = scene_graphs_dict[:self.n_samples]
            image_data_dict = image_data_dict[:self.n_samples]
        logging.info("Processing scene graphs into PyG graphs...")
        
        emb_dim = self.enc_cfg["emb_dim"]
        
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=self.enc_cfg["model_name"], pretrained=self.enc_cfg["pretrained"], device=self.enc_cfg["device"])
        tokenizer = open_clip.get_tokenizer(model_name=self.enc_cfg["model_name"])

        image_id_to_path = dict()
        for dir in [Path(self.raw_dir)/"VG_100K", Path(self.raw_dir)/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                image_id_to_path[img_id] = str(path)

        cached_img_enc_path = osp.join(self.processed_dir, f"{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}_{self.n_samples}_img_enc_cache.pt")
        if not osp.exists(cached_img_enc_path):
            logging.info("Embedding images with CLIP...")
            cached_img_enc = dict()  
            for d in tqdm(scene_graphs_dict):
                img_path = image_id_to_path[d['image_id']]
                with torch.no_grad():
                    img_enc = model.encode_image(preprocess(Image.open(img_path)).unsqueeze(0).to(self.enc_cfg["device"])).cpu()
                    cached_img_enc[img_path] = img_enc
            torch.save(cached_img_enc, cached_img_enc_path)
        cached_img_enc = torch.load(cached_img_enc_path)
        def img_enc_fn(img_path):
            return cached_img_enc[img_path]
        def clip_latent_txt_enc_fn(txts):
            with torch.no_grad():
                return model.encode_text(tokenizer(txts).to(self.enc_cfg["device"])).cpu()
        def clip_embedding_txt_enc(txts):
           with torch.no_grad():
                tokens = tokenizer(txts).to(self.enc_cfg["device"])
                tokens[tokens == 49407] = 0
                tokens = tokens[:, 1:3]
                out =  model.token_embedding(tokens)
                out = out.reshape(len(txts), emb_dim).cpu()
                return out
                
        txt_enc_fn = clip_latent_txt_enc_fn if self.enc_cfg["use_clip_latents"] else clip_embedding_txt_enc
        logging.info("Producing PyG graphs...")
        data_list = [dict_to_pyg_graph(d, img_enc_fn, txt_enc_fn, image_id_to_path, emb_dim, metadata)
                     for d, metadata in tqdm(zip(scene_graphs_dict, image_data_dict))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.info("Collating PyG graphs...")
        data, slices = self.collate(data_list)

        logging.info("Saving PyG graphs...")
        torch.save((data, slices), self.processed_paths[0])