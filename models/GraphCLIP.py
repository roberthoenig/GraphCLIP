import torch
from datasets.visual_genome import VisualGenome
from utils.dataset_utils import dataset_postprocessor
from utils.eval_utils import compute_ranking_metrics_from_features
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from utils.train_utils import contrastive_loss
import logging
import numpy as np
import os.path as osp
import torch.nn as nn

class GNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 256)
        self.conv2 = GCNConv(256, out_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        return x

class GNN2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        # self.conv2 = GATv2Conv(256, 1024, heads=2, concat=False, edge_dim=1024)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        return x

class GraphCLIP():
    def __init__(self, config):
        self.config = config
        
    def train(self):
        # Model
        if "load_checkpoint_path" in self.config["train_args"] and not self.config["train_args"]["load_checkpoint_path"] == "":
            model = torch.load(self.config["train_args"]["load_checkpoint_path"])
        else:
            arch = self.config["model_args"]["architecture"]
            if arch == "GNN":
                model = GNN(**self.config["model_args"]["arch_args"])
            elif arch == "GNN2":
                model = GNN2(**self.config["model_args"]["arch_args"])
            else:
                raise Exception(f"Unknown architecture {arch}.")
        model.to(self.config["device"])
        model.train()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_sz = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"Model size: {model_sz} parameters.")
        # Dataset
        if self.config["dataset"] == "VisualGenome":
            dataset = VisualGenome(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        dataset = dataset_postprocessor(dataset, **self.config["dataset_postprocessor_args"])
        train_val_split = self.config["train_args"]["train_val_split"]
        if train_val_split == "mscoco":
            train_set = dataset_postprocessor(dataset, filter="remove_mscoco_val")
            val_set = dataset_postprocessor(dataset, filter="keep_mscoco_val")
        else:
            train_ratio = train_val_split
            train_set, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
        train_dloader = DataLoader(train_set, batch_size=self.config["train_args"]["batch_size"], shuffle=True)
        val_dloader = DataLoader(val_set, batch_size=self.config["train_args"]["batch_size"], shuffle=False)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["train_args"]["learning_rate"])
        # Training
        pbar_epochs = tqdm(range(self.config["train_args"]["epochs"]), position=0)
        for epoch in pbar_epochs:
            # Train
            train_losses = []
            mov_avg_train_loss = 0
            pbar_train = tqdm(train_dloader, position=1, leave=False)
            for data in pbar_train:
                data = data.to(self.config["device"])
                optimizer.zero_grad()
                loss = contrastive_loss(model(data), data.y, model.logit_scale)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                mov_avg_train_loss = 0.9 * mov_avg_train_loss + 0.1 * loss.item()
                pbar_train.set_postfix({'moving average train_loss': mov_avg_train_loss})
            # Validate
            val_losses = []
            mov_avg_val_loss = 0
            pbar_val = tqdm(val_dloader, position=1, leave=False)
            for data in pbar_val:
                with torch.no_grad():
                    data = data.to(self.config["device"])
                    loss = contrastive_loss(model(data), data.y, model.logit_scale)
                    val_losses.append(loss.item())
                mov_avg_val_loss = 0.9 * mov_avg_val_loss + 0.1 * loss.item()
                pbar_train.set_postfix({'moving average val_loss': mov_avg_val_loss})
            # Log
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            pbar_epochs.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
            logging.info(f"Epoch {epoch}, average train_loss: {train_loss}, average val_loss: {val_loss}")
            # Save Checkpoint
            if (epoch+1) % self.config["train_args"]["epochs_per_checkpoint"] == 0:
                logging.info(f"Saving checkpoint...")
                model.cpu()
                torch.save(model, osp.join(self.config["experiment_dir"], f"checkpoint_{epoch+1}.pt"))
                model.to(self.config["device"])
    
    def eval(self):
        # Model
        model = torch.load(self.config["eval_args"]["load_checkpoint_path"])
        model.to(self.config["device"])
        model.eval()
        
        # Dataset
        if self.config["dataset"] == "VisualGenome":
            dataset = VisualGenome(**self.config["dataset_args"])
            dataset = dataset_postprocessor(dataset, **self.config["dataset_postprocessor_args"])
            train_val_split = self.config["train_args"]["train_val_split"]
            if train_val_split == "mscoco":
                val_set = dataset_postprocessor(dataset, filter="keep_mscoco")
            else:
                train_ratio = train_val_split
            _, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
            val_dloader = DataLoader(val_set, batch_size=1, shuffle=False)
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        
        # Compute features
        with torch.no_grad(), torch.cuda.amp.autocast():
            # (n_samples, emb_sz) 
            logging.info("Computing image embeddings.")
            img_features = torch.concat([model(data.to(self.config["device"])).cpu() for data in tqdm(val_dloader)])
            img_features /= img_features.norm(dim=-1, keepdim=True)
            # (n_samples, captions_per_image, emb_sz) 
            logging.info("Computing caption embeddings.")
            cap_features = torch.concat([data.y.cpu() for data in tqdm(val_dloader)])
            cap_features /= cap_features.norm(dim=-1, keepdim=True)
            cap_features = cap_features.unsqueeze(1)
            
        # Compute metrics
        compute_ranking_metrics_from_features(
            img_features=img_features,
            cap_features=cap_features,
            ks=self.config['eval_args']['ks']
        )