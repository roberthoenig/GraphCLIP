from typing import Any

from .vision_transformer.open_clip.jt_ViT_RelClassifier_lightning import ViT_RelClassifier
from .vision_transformer.open_clip.jt_ViT_RelClassifier_lightning_old import ViT_RelClassifier as ViT_RelClassifier_old
from tempfile import NamedTemporaryFile
import numpy as np
import os
from .download_weights import download_weights
import torch
import logging
import open_clip
from torch_geometric.data import Batch
from .data import FILTERED_RELATIONSHIPS, FILTERED_OBJECTS, FILTERED_ATTRIBUTES

def get_free_gpu(min_mem=9000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            print("Not enough memory on GPU, using CPU")
            return torch.device("cpu")
        return torch.device("cuda", np.argmax(memory_available))
    except:
        print("Could not get free GPU, using CPU")
        return torch.device("cpu")

class Evaluator:
    def __init__(self, evaluator_name, device='auto', model_weights_path=None):
        self.evaluator_name = evaluator_name
        self.weights_name = Evaluator._get_weights_name(evaluator_name)
        self.device = device if device != 'auto' else get_free_gpu()
        print(f'Using device {self.device} for evaluation.')
        if model_weights_path is None:
            model_weights_path = os.path.join(os.path.dirname(__file__), 'data', self.weights_name)
        # check if weights are downloaded, otherwise download them
            if not os.path.exists(model_weights_path):
                download_weights(self.weights_name)
                print(f'Downloaded weights for {self.evaluator_name} to {model_weights_path}.')
        if self.evaluator_name == 'ViT-B/32':
            self._evaluator = ViTBaseLargeEvaluator(device=self.device, model_weights_path=model_weights_path, size='base')
        elif self.evaluator_name == 'ViT-L/14':
            self._evaluator = ViTBaseLargeEvaluator(device=self.device, model_weights_path=model_weights_path, size='large')
        elif self.evaluator_name == 'ViT-L/14-Datacomp':
            self._evaluator = ViTBaseLargeEvaluator(device=self.device, model_weights_path=model_weights_path, size='large')
        elif self.evaluator_name == 'CLIP_ViT-L/14':
            self._evaluator = CLIPEvaluator(device=self.device, model='ViT-L-14', pretrained='laion2b_s32b_b82k')
        elif self.evaluator_name == 'CLIP_ViT-G/14':
            self._evaluator = CLIPEvaluator(device=self.device, model='ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        elif self.evaluator_name == 'histogram':
            self._evaluator = HistogramEvaluator(device=self.device, model_weights_path=model_weights_path)

    def _get_weights_name(evaluator_name):
        if evaluator_name == 'ViT-B/32':
            return 'ViT-Base_Text_Emb_Hockey_Fighter.ckpt'     
        elif evaluator_name == 'ViT-L/14':
            return 'ViT-Large_Text_Emb_Spring_River.ckpt'  
        elif evaluator_name == 'histogram':
            return 'histogram.ckpt'
        elif evaluator_name == 'CLIP_ViT-L/14':
            # CLIP weights are downloaded by open_clip
            return  'dummy.ckpt'
        elif evaluator_name == 'CLIP_ViT-G/14':
            # CLIP weights are downloaded by open_clip
            return  'dummy.ckpt'
        elif evaluator_name == 'GraphCLIP':
            return 'GraphCLIP.ckpt'   
        elif evaluator_name == 'ViT-L/14-Datacomp':
            return 'ViT-Large_Text_Emb_Vocal_Snow.ckpt'
        else:
            raise Exception(f"Unknown evaluator {evaluator_name}.")

    def __call__(self,images, graphs):
        score = self._evaluator(images, graphs)
        return score

class HistogramEvaluator:
    def __init__(self, device, model_weights_path):
        hists = torch.load(model_weights_path)
        self.rel_hist = hists['rel_hist']
        self.attr_hist = hists['attr_hist']
        
    def graph_to_rel_strs(self, graph):
        rel_strs = []
        for sid, oid in list(graph.edges):
            os = graph.nodes[sid]['name']
            oo = graph.nodes[oid]['name']
            rel = graph.edges[(sid, oid)]['predicate']
            rel_str = f'{os} {rel} {oo}'.strip().lower()
            rel_strs.append(rel_str)
        return rel_strs
        
    def graph_to_attr_strs(self, graph):
        attr_strs = []
        for nid in list(graph.nodes):
            node = graph.nodes[nid]
            name = node['name']
            for attr in node.get('attributes', []):
                attr_name_str = f'{attr} {name}'.strip().lower()
                attr_strs.append(attr_name_str)
        return attr_strs
    
    def __call__(self, images, graphs):
        scores = []
        for graph in graphs:
            rel_strs = self.graph_to_rel_strs(graph)
            rel_score = 1
            for rel_str in rel_strs:
                rel_score *= (1+self.rel_hist.get(rel_str, 0))
            attr_strs = self.graph_to_attr_strs(graph)
            attr_score = 1
            for attr_str in attr_strs:
                attr_score *= (1+self.attr_hist.get(attr_str, 0))
            score = rel_score * attr_score
            scores.append(score)
        scores_dict = {
            'overall_scores': scores
        } 
        return scores_dict

class CLIPEvaluator:
    def __init__(self, device, model, pretrained):
        self.device = device
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model)

    def _emb_txts(self, txts):
        txt_embs = []
        for txt in txts:
            tokenized_text = self.tokenizer([txt]).to(self.device)
            txt_emb = self.model.encode_text(tokenized_text).cpu()
            txt_embs.append(txt_emb)
        txt_embs = torch.concat(txt_embs)
        return txt_embs
    
    def _emb_imgs(self, images):
        # images: list of PIL images
        # return: embedding tensor, (length(images), 2048)
        img_embs = []
        for img in images:
            img_emb = self.model.encode_image(self.preprocessor(img).unsqueeze(0).to(self.device)).cpu()
            img_embs.append(img_emb)
        img_embs = torch.concat(img_embs)
        return img_embs
    
    def __call__(self, images, graphs):
        texts = []
        for g in graphs:
            txt = g.caption.split('.')[0]
            texts.append(txt)
        with torch.no_grad():
            # (n_samples, emb_sz) 
            logging.info("Computing image embeddings...")
            txt_embs = self._emb_txts(texts)
            img_embs = self._emb_imgs(images)
            # (n_samples, captions_per_image, emb_sz) 
            logging.info("Computing caption embeddings...")
            txt_embs /= txt_embs.norm(dim=-1, keepdim=True)
            img_embs /= img_embs.norm(dim=-1, keepdim=True)
        scores = torch.sum(txt_embs * img_embs, dim=1)
        scores_dict = {
            'overall_scores': scores.tolist()
        }    
        return scores_dict



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
        if clip_model_type == 'ViT-B/32':
            model = ViT_RelClassifier_old(
                n_rel_classes=100, 
                n_obj_classes=200, 
                n_attr_classes=100, 
                clip_model=clip_model_type, 
                pretrained=clip_pretrained_dataset, 
                shallow=True, 
                mode='text_embeddings'
            )
        elif clip_model_type == 'ViT-L-14':
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

