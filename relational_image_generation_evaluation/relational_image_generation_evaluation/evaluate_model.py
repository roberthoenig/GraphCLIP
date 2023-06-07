from typing import Any

from .graph_clip.data_utils import add_master_node_with_bidirectional_edges, dict_to_pyg_graph, networkx_to_dict
from .vision_transformer.open_clip.jt_ViT_RelClassifier_lightning import ViT_RelClassifier
from .vision_transformer.jt_training import get_free_gpu
import os
from .download_weights import download_weights
import torch
import logging
from tqdm import tqdm
import open_clip
from torch_geometric.data import Batch

class Evaluator:
    def __init__(self, evaluator_name, device='auto'):
        self.evaluator_name = evaluator_name
        self.weights_name = Evaluator._get_weights_name(evaluator_name)
        self.device = device if device != 'auto' else get_free_gpu()
        print(f'Using device {self.device} for evaluation.')
        model_weights_path = os.path.join(os.path.dirname(__file__), 'data', self.weights_name)
        # check if weights are downloaded, otherwise download them
        if not os.path.exists(model_weights_path):
            download_weights(self.weights_name)
            print(f'Downloaded weights for {self.evaluator_name} to {model_weights_path}.')
        if self.evaluator_name == 'ViT-B/32':
            self._evaluator = ViTBaseLargeEvaluator(device=self.device, model_weights_path=model_weights_path, size='base')
        elif self.evaluator_name == 'ViT-L/14':
            self._evaluator = ViTBaseLargeEvaluator(device=self.device, model_weights_path=model_weights_path, size='large')
        elif self.evaluator_name == 'GraphCLIP':
            self._evaluator = GraphCLIPEvaluator(device=self.device, model_weights_path=model_weights_path, **GraphCLIP_config_1)

    def _get_weights_name(evaluator_name):
        if evaluator_name == 'ViT-B/32':
            return 'ViT-Base_Text_Emb_Hockey_Fighter.ckpt'     
        elif evaluator_name == 'ViT-L/14':
            return 'ViT-Large_Text_Emb_Spring_River.ckpt'   
        elif evaluator_name == 'GraphCLIP':
            return 'GraphCLIP.ckpt'   
        else:
            raise Exception(f"Unknown evaluator {evaluator_name}.")

    def __call__(self,images, graphs):
        score = self._evaluator(images, graphs)
        return score







class ViTBaseLargeEvaluator:
    def __init__(self, device, model_weights_path, size='base'):
        self.size = size
        if size == 'base':
            clip_model_type = 'ViT-B/32'
            clip_pretrained_dataset = 'laion400m_e32'
        elif size == 'large':
            clip_model_type = 'ViT-L-14'
            clip_pretrained_dataset = 'laion2b_s32b_b82k'
        self.device = device
        self.model_weights_path = model_weights_path
        model, preprocess = self.load_vit(clip_model_type, clip_pretrained_dataset, device=self.device)
        self.model = model
        self.preprocess = preprocess
        self.rel_classes = {rel:i for i,rel in enumerate(FILTERED_RELATIONSHIPS)}
        self.obj_classes = {obj:i for i,obj in enumerate(FILTERED_OBJECTS)}
        self.attr_classes = {attr:i for i,attr in enumerate(FILTERED_ATTRIBUTES)}
        obj_embeddings = torch.load(os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt'), map_location=self.device)
        self.text_embeddings = {obj:torch.tensor(obj_embeddings[i]) for i,obj in enumerate(FILTERED_OBJECTS)}

    def load_vit(self,clip_model_type, clip_pretrained_dataset, device):
        model = ViT_RelClassifier(
            n_rel_classes=100, 
            n_obj_classes=200, 
            n_attr_classes=100, 
            clip_model=clip_model_type, 
            pretrained=clip_pretrained_dataset, 
            shallow=True, 
            mode='text_embeddings'
        )
        prepocess_function = model.preprocess
        model.to(device)
        loaded = torch.load(self.model_weights_path, map_location=device)
        model.load_state_dict(loaded['state_dict'])
        model.eval()
        return model, prepocess_function
    
    def __call__(self, images, graphs):
        # images: list of PIL images
        # graphs: list of networkx graphs
        # returns: dict of list of scores, with two lists: 'rel_scores' and 'attr_scores'

        # preprocess images
        images = [img.convert("RGB") for img in images] # convert to RGB, not entirely sure if this is necessary, but copilot suggested it and it was trained like this
        images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        rel_confidences = []
        for image, g in zip(images, graphs):
            g_rel_confidences = []
            for edge in g.edges:
                text_embd_obj1 = self.text_embeddings[g.nodes[edge[0]]['name']].to(self.device)
                text_embd_obj2 = self.text_embeddings[g.nodes[edge[1]]['name']].to(self.device)
                full_text_clip_embd = torch.cat((text_embd_obj1, text_embd_obj2), dim=0).reshape(1,-1)
                rel_label = torch.tensor(self.rel_classes[g.edges[edge]['predicate']])
                if self.size == 'base':
                    rel, obj1, obj2, attr = self.model(image.unsqueeze(0), full_text_clip_embd.unsqueeze(0))
                elif self.size == 'large':
                    rel, obj1, obj2, attr, _ = self.model(image.unsqueeze(0), full_text_clip_embd.unsqueeze(0))
                else:
                    raise ValueError('size must be either "base" or "large"')
                rel = torch.softmax(rel, dim=1)
                g_rel_confidences.append(rel[0][rel_label].item())
            if len(g_rel_confidences) > 0:
                rel_confidences.append(torch.mean(torch.tensor(g_rel_confidences)).cpu().item())
            else:
                rel_confidences.append('noedges')
        attr_confidences = []
        for image, g in zip(images, graphs):
            g_attr_confidences = []
            for node in g.nodes:
                text_embd_obj = self.text_embeddings[g.nodes[node]['name']].to(self.device)
                full_text_clip_embd = torch.cat((text_embd_obj, text_embd_obj), dim=0).reshape(1,-1)
                obj_label = torch.tensor(self.obj_classes[g.nodes[node]['name']])
                if self.size == 'base':
                    rel, obj1, obj2, attr = self.model(image.unsqueeze(0), full_text_clip_embd.unsqueeze(0))
                elif self.size == 'large':
                    rel, obj1, obj2, attr1, attr2 = self.model(image.unsqueeze(0), full_text_clip_embd.unsqueeze(0))
                    # mean of attr1 and attr2
                    attr = (attr1 + attr2) / 2
                else:
                    raise ValueError('size must be either "base" or "large"')
                attr = torch.sigmoid(attr)
                attr_labels = [self.attr_classes[attr] for attr in g.nodes[node]['attributes']]
                for attr_label in attr_labels:
                    g_attr_confidences.append(attr[0][attr_label].item())
            if len(g_attr_confidences) > 0:
                attr_confidences.append(torch.mean(torch.tensor(g_attr_confidences)).cpu().item())
            else:
                attr_confidences.append('noattributes')
        return {'rel_scores': rel_confidences, 'attr_scores': attr_confidences}

GraphCLIP_config_1 = {
    'CLIP_cfg': {
        'model_name': "ViT-g-14",
        'pretrained': "laion2b_s12b_b42k",
    },
    'normalize': False,
    'use_long_rel_enc': False,
    'transform': "add_master_node_with_bidirectional_edges"
}
class GraphCLIPEvaluator:
    def __init__(self, device, model_weights_path, CLIP_cfg, normalize, use_long_rel_enc, transform):
        self.device = device
        self.normalize = normalize
        self.model_weights_path = model_weights_path
        self.tokenizer = open_clip.get_tokenizer(model_name=CLIP_cfg["model_name"])
        self.img_model, _, self.img_preprocess = open_clip.create_model_and_transforms(model_name=CLIP_cfg["model_name"], pretrained=CLIP_cfg["pretrained"], device=self.device)
        self.graph_model = self.load_graph_model()
        self.use_long_rel_enc = use_long_rel_enc
        self.transform = transform

    def load_graph_model(self):
        model = torch.load(self.model_weights_path)
        model.to(self.device)
        model.eval()
        return model
    
    def _txt_enc(self, txts):
        with torch.no_grad():
            tokens = self.tokenizer(txts)
            tokens[tokens == 49407] = 0
            tokens = tokens[:, 1:3]
            out = tokens.cpu()
            return out  
    
    def _emb_imgs(self, images):
        # images: list of PIL images
        # return: embedding tensor, (length(images), 2048)
        img_embs = []
        for img in tqdm(images):
            img_emb = self.img_model.encode_image(self.img_preprocess(img).unsqueeze(0).to(self.device)).cpu()
            img_embs.append(img_emb)
        img_embs = torch.concat(img_embs)
        return img_embs
        
    def _emb_graphs(self, graphs):
        # graphs: list of networkx graphs
        # return: embedding tensor, (length(graphs), 2048)
        graph_embs = []
        for graph in tqdm(graphs):
            d = networkx_to_dict(graph)
            data = dict_to_pyg_graph(d, txt_enc=self._txt_enc, use_long_rel_enc=self.use_long_rel_enc)
            if self.transform == "add_master_node_with_bidirectional_edges":
                data = add_master_node_with_bidirectional_edges(data)
            else:
                raise Exception(f"Unknown transform {self.transform}")
            batch = Batch.from_data_list([data])
            graph_emb = self.graph_model(batch.to(self.device)).cpu()
            graph_embs.append(graph_emb)
        graph_embs = torch.concat(graph_embs)
        return graph_embs
    
    def __call__(self, images, graphs):
        # images: list of PIL images
        # graphs: list of networkx graphs
        # returns: list of scores (only guaranteed to be between [-1, 1] if normalize=True)   
        # Compute features
        with torch.no_grad():
            # (n_samples, emb_sz) 
            logging.info("Computing image embeddings...")
            cap_embs = self._emb_graphs(graphs)
            img_embs = self._emb_imgs(images)
            # (n_samples, captions_per_image, emb_sz) 
            logging.info("Computing caption embeddings...")
            cap_embs = cap_embs.unsqueeze(1)
            if self.normalize:
                cap_embs /= cap_embs.norm(dim=-1, keepdim=True)
                img_embs /= img_embs.norm(dim=-1, keepdim=True)
        scores = torch.sum(cap_embs * img_embs, dim=1).tolist()
        return scores    

                                                                              





FILTERED_OBJECTS = ['man', 'person', 'window', 'tree', 'building', 'shirt', 'wall', 'woman', 'sign', 'sky', 'ground', 'grass', 'table', 'pole', 'head', 'light', 'water', 'car', 'hand', 'hair', 'people', 'leg', 'trees', 'clouds', 'ear', 'plate', 'leaves', 'fence', 'door', 'pants', 'eye', 'train', 'chair', 'floor', 'road', 'street', 'hat', 'snow', 'wheel', 'shadow', 'jacket', 'nose', 'boy', 'line', 'shoe', 'clock', 'sidewalk', 'boat', 'tail', 'cloud', 'handle', 'letter', 'girl', 'leaf', 'horse', 'bus', 'helmet', 'bird', 'giraffe', 'field', 'plane', 'flower', 'elephant', 'umbrella', 'dog', 'shorts', 'arm', 'zebra', 'face', 'windows', 'sheep', 'glass', 'bag', 'cow', 'bench', 'cat', 'food', 'bottle', 'rock', 'tile', 'kite', 'tire', 'post', 'number', 'stripe', 'surfboard', 'truck', 'logo', 'glasses', 'roof', 'skateboard', 'motorcycle', 'picture', 'flowers', 'bear', 'player', 'foot', 'bowl', 'mirror', 'background', 'pizza', 'bike', 'shoes', 'spot', 'tracks', 'pillow', 'shelf', 'cap', 'mouth', 'box', 'jeans', 'dirt', 'lights', 'legs', 'house', 'part', 'trunk', 'banana', 'top', 'plant', 'cup', 'counter', 'board', 'bed', 'wave', 'bush', 'ball', 'sink', 'button', 'lamp', 'beach', 'brick', 'flag', 'neck', 'sand', 'vase', 'writing', 'wing', 'paper', 'seat', 'lines', 'reflection', 'coat', 'child', 'toilet', 'laptop', 'airplane', 'letters', 'glove', 'vehicle', 'phone', 'book', 'branch', 'sunglasses', 'edge', 'cake', 'desk', 'rocks', 'frisbee', 'tie', 'tower', 'animal', 'hill', 'mountain', 'headlight', 'ceiling', 'cabinet', 'eyes', 'stripes', 'wheels', 'lady', 'ocean', 'racket', 'container', 'skier', 'keyboard', 'towel', 'frame', 'windshield', 'hands', 'back', 'track', 'bat', 'finger', 'pot', 'orange', 'fork', 'waves', 'design', 'feet', 'basket', 'fruit', 'broccoli', 'engine', 'guy', 'knife', 'couch', 'railing', 'collar', 'cars']
FILTERED_RELATIONSHIPS = ['on', 'has', 'in', 'of', 'wearing', 'with', 'behind', 'holding', 'on a', 'near', 'on top of', 'next to', 'has a', 'under', 'of a', 'by', 'above', 'wears', 'in front of', 'sitting on', 'on side of', 'attached to', 'wearing a', 'in a', 'over', 'are on', 'at', 'for', 'around', 'beside', 'standing on', 'riding', 'standing in', 'inside', 'have', 'hanging on', 'walking on', 'on front of', 'are in', 'hanging from', 'carrying', 'holds', 'covering', 'belonging to', 'between', 'along', 'eating', 'and', 'sitting in', 'watching', 'below', 'painted on', 'laying on', 'against', 'playing', 'from', 'inside of', 'looking at', 'with a', 'parked on', 'to', 'has an', 'made of', 'covered in', 'mounted on', 'says', 'growing on', 'across', 'part of', 'on back of', 'flying in', 'outside', 'lying on', 'worn by', 'walking in', 'sitting at', 'printed on', 'underneath', 'crossing', 'beneath', 'full of', 'using', 'filled with', 'hanging in', 'covered with', 'built into', 'standing next to', 'adorning', 'a', 'in middle of', 'flying', 'supporting', 'touching', 'next', 'swinging', 'pulling', 'growing in', 'sitting on top of', 'standing', 'lying on top of']
FILTERED_ATTRIBUTES = ['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'wooden', 'gray', 'silver', 'metal', 'orange', 'grey', 'tall', 'long', 'dark', 'pink', 'clear', 'standing', 'round', 'tan', 'glass', 'here', 'wood', 'open', 'purple', 'big', 'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young', 'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy', 'colorful', 'one', 'beige', 'bare', 'wet', 'light', 'square', 'little', 'closed', 'stone', 'blonde', 'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted', 'thick', 'part', 'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular', 'looking', 'grassy', 'dry', 'light brown', 'cement', 'leafy', 'wearing', 'tiled', "man's", 'light blue', 'baseball', 'cooked', 'pictured', 'curved', 'decorative', 'dead', 'eating', 'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt', 'watching', 'colored', 'stuffed', 'circular']