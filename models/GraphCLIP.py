import torch
from PIL import Image
import open_clip
from datasets.mscoco import MSCOCO
from datasets.visual_genome import VisualGenome
from utils.eval_utils import compute_ranking_metrics_from_features
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from utils.train_utils import contrastive_loss
import logging
import numpy as np
import os.path as osp

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1024, 256)
        self.conv2 = GCNConv(256, 1024)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
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
            model = GNN()
        model.to(self.config["device"])
        model.train()
        # Dataset
        if self.config["dataset"] == "VisualGenome":
            dataset = VisualGenome(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        train_ratio = self.config["train_args"]["train_val_split"]
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
                loss = contrastive_loss(model(data), data.y)
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
                    loss = contrastive_loss(model(data), data.y)
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
        raise NotImplementedError