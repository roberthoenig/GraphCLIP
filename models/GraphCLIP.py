import torch
from datasets.visual_genome import VisualGenome, VisualGenomeAdversarial
from models.MyLayer import construct_my_layer
from models.MyTransformerConv import MyTransformerConv
from utils.dataset_utils import dataset_filter, make_sample_relation_batched, transfer_attributes_batched, tokens_to_embeddings_batched
from utils.eval_utils import compute_ranking_metrics_from_features, compute_accuracy_from_adversarial_features
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_node
from utils.train_utils import contrastive_adv_loss, contrastive_loss
from utils.model_utils import dropout_node_keep_master_nodes, global_master_pool
import logging
import numpy as np
import os.path as osp
import torch.nn as nn
import open_clip

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

# Like GNN2, but works with master node
class GNN3(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_master_pool(x, batch)
        return x

# Like GNN3, but uses dropout
class GNN4(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _, _ = dropout_node(edge_index, training=self.training, p=self.p_dropout)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_master_pool(x, batch)
        return x

# Like GNN3, but uses dropout in GATv2Conv
class GNN5(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim, dropout=p_dropout)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim, dropout=p_dropout)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim, dropout=p_dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_master_pool(x, batch)
        return x

# Like GNN4, but accepts graphs with tokenized (i.e. not yet embedded) features.
class GNN6(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights =  torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _, _ = dropout_node(edge_index, training=self.training, p=self.p_dropout)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_master_pool(x, batch)
        return x

# Like GNN6, but actually uses edge attributes (◔_◔).
class GNN7(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights =  torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, _ = dropout_node(edge_index, training=self.training, p=self.p_dropout)
        edge_attr = edge_attr[edge_mask]
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        return x

# Like GNN7, but fixes dropout to not drop the master node.
class GNN8(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, _ = dropout_node_keep_master_nodes(edge_index=edge_index, batch=batch, training=self.training, p=self.p_dropout)
        edge_attr = edge_attr[edge_mask]
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        return x

# Like GNN8, but supports adversarial relation sampling.
class GNN9(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=self.p_dropout, exclude_from_dropout=exclude_from_dropout,
                                                                  dropout_mask=dropout_mask)
        edge_attr = edge_attr[edge_mask]
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        if return_dropout_mask:
            return x, node_mask
        else:
            return x

# Like GNN9, but with edge dimension lowering.
class GNN10(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, edge_projected_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.conv2 = GATv2Conv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.conv3 = GATv2Conv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        self.project_edges = torch.nn.Linear(edge_dim, edge_projected_dim)
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=self.p_dropout, exclude_from_dropout=exclude_from_dropout,
                                                                  dropout_mask=dropout_mask)
        edge_attr = edge_attr[edge_mask]
        edge_attr = self.project_edges(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        if return_dropout_mask:
            return x, node_mask
        else:
            return x

# Like GNN10, but uses TransformerConv instead of GATv2Conv.
class GNN11(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, edge_projected_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init):
        super().__init__()
        self.conv1 = MyTransformerConv(in_dim, middle_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.conv2 = MyTransformerConv(middle_dim, middle_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.conv3 = MyTransformerConv(middle_dim, out_dim, heads=2, concat=False, edge_dim=edge_projected_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        self.project_edges = torch.nn.Linear(edge_dim, edge_projected_dim)
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=self.p_dropout, exclude_from_dropout=exclude_from_dropout,
                                                                  dropout_mask=dropout_mask)
        edge_attr = edge_attr[edge_mask]
        edge_attr = self.project_edges(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        if return_dropout_mask:
            return x, node_mask
        else:
            return x
        
# Like GNN10, but MetaConv layer.
class GNN12(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, edge_projected_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init, zero_edge_attr=False):
        super().__init__()
        self.conv1 = construct_my_layer(node_in_dim=in_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv2 = construct_my_layer(node_in_dim=middle_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv3 = construct_my_layer(node_in_dim=middle_dim, node_out_dim=out_dim, edge_in_dim=edge_projected_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        self.project_edges = torch.nn.Linear(edge_dim, edge_projected_dim)
        self.zero_edge_attr = zero_edge_attr
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            new_embs = torch.sin(torch.arange(4, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=self.p_dropout, exclude_from_dropout=exclude_from_dropout,
                                                                  dropout_mask=dropout_mask)
        if self.zero_edge_attr:
            edge_attr[True] = 0
        edge_attr = edge_attr[edge_mask]
        edge_attr = self.project_edges(edge_attr)
        x, edge_attr, _ = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x, edge_attr, _ = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x, edge_attr, _ = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        if return_dropout_mask:
            return x, node_mask
        else:
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
            elif arch == "GNN3":
                model = GNN3(**self.config["model_args"]["arch_args"])
            elif arch == "GNN4":
                model = GNN4(**self.config["model_args"]["arch_args"])
            elif arch == "GNN5":
                model = GNN5(**self.config["model_args"]["arch_args"])
            elif arch == "GNN6":
                model = GNN6(**self.config["model_args"]["arch_args"])
            elif arch == "GNN7":
                model = GNN7(**self.config["model_args"]["arch_args"])
            elif arch == "GNN8":
                model = GNN8(**self.config["model_args"]["arch_args"])
            elif arch == "GNN9":
                model = GNN9(**self.config["model_args"]["arch_args"])
            elif arch == "GNN10":
                model = GNN10(**self.config["model_args"]["arch_args"])
            elif arch == "GNN11":
                model = GNN11(**self.config["model_args"]["arch_args"])
            elif arch == "GNN12":
                model = GNN12(**self.config["model_args"]["arch_args"])
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
        txt_enc = dataset.clip_embedding_txt_enc
        dataset = dataset_filter(dataset, **self.config["dataset_filter_args"])
        train_val_split = self.config["train_args"]["train_val_split"]
        if train_val_split == "mscoco":
            train_set = dataset_filter(dataset, filters=["remove_mscoco_val"])
            val_set = dataset_filter(dataset, filters=["keep_mscoco_val"])
        else:
            train_ratio = train_val_split
            train_set, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
        val_set = dataset_filter(val_set, **self.config["valset_filter_args"])
        train_dloader = DataLoader(train_set, batch_size=self.config["train_args"]["batch_size"], shuffle=True)
        val_dloader = DataLoader(val_set, batch_size=self.config["train_args"]["batch_size"], shuffle=False)
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["train_args"]["learning_rate"])
        # Adversarial transform
        adv_transform = self.config["train_args"].get("adv_transform", None)
        if adv_transform == "transfer_attributes":
            adv_transform = transfer_attributes_batched
        if adv_transform == "sample_relation":
            adv_transform = make_sample_relation_batched(txt_enc, **self.config['train_args'].get('adv_transform_args', dict()))
        elif adv_transform is not None:
            logging.info(f"Unknown adversarial transform {adv_transform}.")

        # Training
        pbar_epochs = tqdm(range(self.config["train_args"]["epochs"]), position=0)
        smallest_val_loss = np.inf
        for epoch in pbar_epochs:
            # Train
            model.train()
            train_losses = []
            mov_avg_train_loss = 0
            pbar_train = tqdm(train_dloader, position=1, leave=False)
            for data in pbar_train:
                optimizer.zero_grad()
                if adv_transform is not None:
                    adv_data = adv_transform(data.clone())
                    adv_data = adv_data.to(self.config["device"])
                    data = data.to(self.config["device"])
                    if self.config['train_args']['exclude_adv_affected_nodes_from_dropout']:
                        exclude_from_dropout = adv_data.adv_affected_nodes
                    else:
                        exclude_from_dropout = None
                    y_adv, dropout_mask = model(adv_data, exclude_from_dropout=exclude_from_dropout, return_dropout_mask=True)
                    y_pred = model(data, dropout_mask=dropout_mask)
                    loss = contrastive_adv_loss(y_pred, y_adv, data.y, model.logit_scale)
                else:
                    data = data.to(self.config["device"])
                    loss = contrastive_loss(model(data), data.y, model.logit_scale)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                mov_avg_train_loss = 0.9 * mov_avg_train_loss + 0.1 * loss.item()
                pbar_train.set_postfix({'moving average train_loss': mov_avg_train_loss})
            # Validate
            model.eval()
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
            cp = self.config["train_args"]["epochs_per_checkpoint"]
            if isinstance(cp, float):
                if val_loss < min(smallest_val_loss, cp):
                    do_cp = True
                    smallest_val_loss = val_loss
                else:
                    do_cp = False
            else:
                do_cp = (epoch+1) % self.config["train_args"]["epochs_per_checkpoint"] == 0
            if do_cp:
                logging.info(f"Saving checkpoint...")
                model.cpu()
                torch.save(model, osp.join(self.config["experiment_dir"], f"checkpoint_{epoch+1}_{val_loss:.2f}.pt"))
                model.to(self.config["device"])
    
    def eval(self):
        # Model
        model = torch.load(self.config["eval_args"]["load_checkpoint_path"])
        model.to(self.config["device"])
        model.eval()
        
        # Dataset
        if self.config["dataset"] == "VisualGenome":
            dataset = VisualGenome(**self.config["dataset_args"])
            dataset = dataset_filter(dataset, **self.config["dataset_filter_args"])
            train_val_split = self.config["eval_args"]["train_val_split"]
            if train_val_split == "mscoco":
                val_set = dataset_filter(dataset, filters=["keep_mscoco_val"])
            else:
                train_ratio = train_val_split
                _, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
            val_set = dataset_filter(val_set, **self.config["valset_filter_args"])
            val_dloader = DataLoader(val_set, batch_size=1, shuffle=False)
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        
        # Compute features
        with torch.no_grad():
            # (n_samples, emb_sz) 
            logging.info("Computing image embeddings.")
            img_features = torch.concat([model(data.to(self.config["device"])).cpu() for data in tqdm(val_dloader)])
            # (n_samples, captions_per_image, emb_sz) 
            logging.info("Computing caption embeddings.")
            cap_features = torch.concat([data.y.cpu() for data in tqdm(val_dloader)])
            cap_features = cap_features.unsqueeze(1)
            if self.config["eval_args"]["normalize"]:
                cap_features /= cap_features.norm(dim=-1, keepdim=True)
                img_features /= img_features.norm(dim=-1, keepdim=True)
            
        # Compute metrics
        compute_ranking_metrics_from_features(
            img_features=img_features,
            cap_features=cap_features,
            ks=self.config['eval_args']['ks']
        )

    def eval_adversarial(self):
        # Model
        model = torch.load(self.config["eval_args"]["load_checkpoint_path"])
        model.to(self.config["device"])
        model.eval()
        
        # Dataset
        if self.config["dataset"] == "VisualGenomeAdversarial":
            dataset = VisualGenomeAdversarial(**self.config["dataset_args"])
            dataset = dataset_filter(dataset, **self.config["dataset_filter_args"])
            train_val_split = self.config["eval_args"]["train_val_split"]
            train_ratio = train_val_split
            _, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
            val_set = dataset_filter(val_set, **self.config["valset_filter_args"])
            val_dloader = DataLoader(val_set, batch_size=1, shuffle=False)
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        # Compute features
        with torch.no_grad():
            # (n_samples, emb_sz) 
            logging.info("Computing groundtruth graph embeddings...")
            features_gt = torch.concat([model(sample["gt"].to(self.config["device"])).cpu() for sample in tqdm(val_dloader)])
            # (n_samples, emb_sz) 
            logging.info("Computing adversarial graph embeddings...")
            features_adv = torch.concat([model(sample["adv"].to(self.config["device"])).cpu() for sample in tqdm(val_dloader)])
            # (n_samples, emb_sz) 
            logging.info("Retrieving image embeddings...")
            # Note: sample["gt"].y should be identical to sample["adv"].y
            img_features = torch.concat([sample["gt"].y.cpu() for sample in tqdm(val_dloader)])
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