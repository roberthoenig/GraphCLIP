import torch
from PIL import Image
import open_clip
from datasets.mscoco import MSCOCO
from datasets.visual_genome import VisualGenomeAdversarialText
from utils.eval_utils import compute_accuracy_from_adversarial_features, compute_ranking_metrics_from_features
from tqdm import tqdm
import logging
from pprint import pprint

class TextCLIP():
    def __init__(self, config):
        self.config = config
        
    def eval(self):
        # Model
        model_name = self.config["model_args"]["model_name"]
        pretrained = self.config["model_args"]["pretrained"]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=self.config["device"])
        tokenizer = open_clip.get_tokenizer(model_name=model_name)
        
        # Dataset
        if self.config["dataset"] == "MSCOCO":
            dataset = MSCOCO(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        
        # Compute features
        with torch.no_grad(), torch.cuda.amp.autocast():
            # (n_samples, emb_sz) 
            print("Computing image embeddings.")
            img_features = torch.concat([model.encode_image(preprocess(Image.open(path)).to(self.config["device"]).unsqueeze(0)).cpu() for path in tqdm(dataset.img_paths)])
            # (n_samples, captions_per_image, emb_sz) 
            print("Computing caption embeddings.")
            cap_features = torch.stack([model.encode_text(tokenizer(captions).to(self.config["device"])).cpu() for captions in tqdm(dataset.captions)])
            img_features /= img_features.norm(dim=-1, keepdim=True)
            cap_features /= cap_features.norm(dim=-1, keepdim=True)

        # Compute metrics
        compute_ranking_metrics_from_features(
            img_features=img_features,
            cap_features=cap_features,
            ks=self.config['eval_args']['ks']
        )

    def eval_adversarial(self):
        # Model
        model_name = self.config["model_args"]["model_name"]
        pretrained = self.config["model_args"]["pretrained"]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=self.config["device"])
        tokenizer = open_clip.get_tokenizer(model_name=model_name)
        
        # Dataset
        if self.config["dataset"] == "VisualGenomeAdversarialText":
            dataset = VisualGenomeAdversarialText(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        
        # Compute features
        with torch.no_grad(), torch.cuda.amp.autocast():
            # (n_samples, emb_sz) 
            logging.info("Computing groundtruth text embeddings...")
            features_gt = torch.concat([model.encode_text(tokenizer(c).to(self.config["device"])).cpu() for c in tqdm(dataset.captions_gt)])
            # (n_samples, emb_sz) 
            logging.info("Computing adversarial text embeddings...")
            features_adv = torch.concat([model.encode_text(tokenizer(c).to(self.config["device"])).cpu() for c in tqdm(dataset.captions_adv)])
            pprint(list(zip(dataset.captions_gt, dataset.captions_adv, dataset.img_paths)))
            # (n_samples, emb_sz) 
            logging.info("Retrieving image embeddings...")
            img_features = torch.concat([model.encode_image(preprocess(Image.open(path)).to(self.config["device"]).unsqueeze(0)).cpu() for path in tqdm(dataset.img_paths)])
            if self.config["eval_args"]["normalize"]:
                img_features /= img_features.norm(dim=-1, keepdim=True)
                features_gt /= features_gt.norm(dim=-1, keepdim=True)
                features_adv /= features_adv.norm(dim=-1, keepdim=True)
            
        # Compute metrics
        compute_accuracy_from_adversarial_features(
            img_features=img_features,
            features_gt=features_gt,
            features_adv=features_adv,
        )