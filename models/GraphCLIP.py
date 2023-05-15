import torch
from datasets.visual_genome import VisualGenome, VisualGenomeAdversarial, VisualGenomeAdversarial2, VisualGenomeAdversarialAttr
from models.MyLayer import construct_my_layer, construct_my_layer2
from models.MyTransformerConv import MyTransformerConv
from utils.dataset_utils import MultiDataLoader, dataset_filter, make_sample_all_relations_batched, make_sample_relation_batched, transfer_attributes_batched, tokens_to_embeddings_batched
from utils.eval_utils import compute_ranking_metrics_from_features, compute_accuracy_from_adversarial_features
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_node
from utils.train_utils import binary_adv_crossentropy_loss, contrastive_adv_loss, contrastive_loss
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
        if self.zero_edge_attr:
            edge_attr[True] = 0
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
            avg_norm = model.token_embedding.weight.norm(dim=1).mean()
            new_embs = torch.sin(torch.arange(1, 5, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            new_embs_norm = new_embs.norm(dim=1).mean()
            new_embs = new_embs * (avg_norm/new_embs_norm)
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
        
# Like GNN12, but MetaConv layer that uses unnormalized features for the attention contribution from edges.
class GNN13(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, edge_projected_dim, middle_dim, p_dropout, model_name, pretrained, freeze_embedding, embedding_init, zero_edge_attr=False):
        super().__init__()
        self.edge_projected_dim = edge_projected_dim
        self.conv1 = construct_my_layer2(node_in_dim=in_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv2 = construct_my_layer2(node_in_dim=middle_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv3 = construct_my_layer2(node_in_dim=middle_dim, node_out_dim=out_dim, edge_in_dim=edge_projected_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.p_dropout = p_dropout
        self.project_edges = torch.nn.Linear(edge_dim, edge_projected_dim)
        self.project_edges2 = torch.nn.Linear(edge_dim, edge_projected_dim)
        self.zero_edge_attr = zero_edge_attr
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            avg_norm = model.token_embedding.weight.norm(dim=1).mean()
            new_embs = torch.sin(torch.arange(1, 5, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            new_embs2 = new_embs
            new_embs_norm = new_embs.norm(dim=1).mean()
            new_embs = new_embs * (avg_norm/new_embs_norm)
            weights = torch.cat([model.token_embedding.weight, new_embs])
            weights2 = torch.cat([model.token_embedding.weight, new_embs2])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)
        self.embedding2 = torch.nn.Embedding.from_pretrained(weights2, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None):
        data_clone = data.clone()
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr2 = tokens_to_embeddings_batched(data_clone, self.embedding2).edge_attr
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=self.p_dropout, exclude_from_dropout=exclude_from_dropout,
                                                                  dropout_mask=dropout_mask)
        if self.zero_edge_attr:
            edge_attr[True] = 0
        edge_attr = edge_attr[edge_mask]
        edge_attr = self.project_edges(edge_attr)
        edge_attr2 = edge_attr2[edge_mask]
        edge_attr2 = self.project_edges2(edge_attr2)
        x, edge_attr, _ = self.conv1(x, edge_index, torch.cat([edge_attr, edge_attr2], dim=1))
        x = F.relu(x)
        edge_attr = F.relu(edge_attr[:, :self.edge_projected_dim])
        x, edge_attr, _ = self.conv2(x, edge_index, torch.cat([edge_attr, edge_attr2], dim=1))
        x = F.relu(x)
        edge_attr = F.relu(edge_attr[:, :self.edge_projected_dim])
        x, edge_attr, _ = self.conv3(x, edge_index, torch.cat([edge_attr, edge_attr2], dim=1))
        x = global_master_pool(x, batch)
        if return_dropout_mask:
            return x, node_mask
        else:
            return x
        
        
# Like GNN12, but moves p_dropout to forward pass.
class GNN14(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, edge_projected_dim, middle_dim, model_name, pretrained, freeze_embedding, embedding_init, zero_edge_attr=False):
        super().__init__()
        self.conv1 = construct_my_layer(node_in_dim=in_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv2 = construct_my_layer(node_in_dim=middle_dim, node_out_dim=middle_dim, edge_in_dim=edge_projected_dim)
        self.conv3 = construct_my_layer(node_in_dim=middle_dim, node_out_dim=out_dim, edge_in_dim=edge_projected_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.project_edges = torch.nn.Linear(edge_dim, edge_projected_dim)
        self.zero_edge_attr = zero_edge_attr
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device="cpu")
        emb_dim = model.token_embedding.embedding_dim
        if embedding_init == 'random':
            new_shape = list(model.token_embedding.weight.shape)
            new_shape[0] += 4
            weights = torch.randn(new_shape)
        elif embedding_init == 'CLIP':
            avg_norm = model.token_embedding.weight.norm(dim=1).mean()
            new_embs = torch.sin(torch.arange(1, 5, dtype=torch.float).reshape(-1,1) * torch.arange(emb_dim, dtype=torch.float).reshape(1,-1))
            new_embs_norm = new_embs.norm(dim=1).mean()
            new_embs = new_embs * (avg_norm/new_embs_norm)
            weights = torch.cat([model.token_embedding.weight, new_embs])
        else:
            raise Exception(f"Unknown embedding_init {embedding_init}.")
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_embedding)

    def forward(self, data, exclude_from_dropout=None, return_dropout_mask=False, dropout_mask=None, p_dropout=None):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_index, edge_mask, node_mask = dropout_node_keep_master_nodes(edge_index=edge_index,
                                                                  batch=batch, training=self.training,
                                                                  p=p_dropout, exclude_from_dropout=exclude_from_dropout,
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
        if "load_checkpoint_path" in self.config.get("train_args", []) and not self.config["train_args"]["load_checkpoint_path"] == "":
            model = torch.load(self.config["train_args"]["load_checkpoint_path"])
        elif "load_checkpoint_path" in self.config and not self.config["load_checkpoint_path"] == "":
            model = torch.load(self.config["load_checkpoint_path"])
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
            elif arch == "GNN13":
                model = GNN13(**self.config["model_args"]["arch_args"])
            elif arch == "GNN14":
                model = GNN14(**self.config["model_args"]["arch_args"])
            else:
                raise Exception(f"Unknown architecture {arch}.")
        model.to(self.config["device"])
        model.train()
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_sz = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"Model size: {model_sz} parameters.")
        
        def config_to_trainset_valset_batchsize_advtransform(cfg):
            # Dataset
            if cfg["dataset_args"]["dataset"] == "VisualGenome":
                dataset = VisualGenome(**cfg["dataset_args"])
            else:
                raise Exception(f"Unkown dataset {cfg['dataset_args']['dataset']}.")
            txt_enc = dataset.clip_embedding_txt_enc
            dataset = dataset_filter(dataset, **cfg["dataset_filter_args"])
            train_val_split = cfg["train_args"]["train_val_split"]
            if train_val_split == "mscoco":
                train_set = dataset_filter(dataset, filters=["remove_mscoco_val"])
                val_set = dataset_filter(dataset, filters=["keep_mscoco_val"])
            else:
                train_ratio = train_val_split
                train_set, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
            val_set = dataset_filter(val_set, **cfg["valset_filter_args"])
             # Adversarial transform
            adv_transform = cfg["train_args"].get("adv_transform", None)
            if adv_transform == "transfer_attributes":
                adv_transform = transfer_attributes_batched
            if adv_transform == "sample_relation":
                adv_transform = make_sample_relation_batched(txt_enc, **cfg['train_args'].get('adv_transform_args', dict()))
            if adv_transform == "replace_all_edges":
                adv_transform = make_sample_all_relations_batched(txt_enc, **cfg['train_args'].get('adv_transform_args', dict()))
            elif adv_transform is not None:
                logging.info(f"Unknown adversarial transform {adv_transform}.")
            # Adversarial transform dropout exclusion
            excl_from_dropout = cfg['train_args'].get('exclude_adv_affected_nodes_from_dropout', None)
            # Loss
            loss_fn_str = cfg["train_args"].get("loss", "contrastive_loss")
            if loss_fn_str == "contrastive_loss":
                loss_fn = contrastive_loss
            elif loss_fn_str == "contrastive_adv_loss":
                loss_fn = contrastive_adv_loss
            elif loss_fn_str == "binary_adv_crossentropy_loss":
                loss_fn = binary_adv_crossentropy_loss
            else:
                raise Exception(f"Unknown loss function {loss_fn_str}.")
            # Checkpointing
            cp = cfg["train_args"]["epochs_per_checkpoint"]
            # Weight
            weight = cfg["train_args"].get("weight", 1.0)
            # Dropout
            p_dropout = cfg["train_args"].get("p_dropout", None)
            return train_set, val_set, cfg["train_args"]["batch_size"], adv_transform, excl_from_dropout, loss_fn, cp, weight, p_dropout
        # Streamline multitask and single task config files.
        if "multitasks" in self.config:
            learning_rate = self.config["learning_rate"]
            train_sets = []
            batch_sizes = []
            adv_transforms = []
            val_dloaders = []
            loss_fns = []
            cps = []
            excl_from_dropouts = []
            weights = []
            p_dropouts = []
            for cfg in self.config["multitasks"]:
                train_set, val_set, batch_size, adv_transform, excl_from_dropout, loss_fn, cp, weight, p_dropout = config_to_trainset_valset_batchsize_advtransform(cfg)
                train_sets.append(train_set)
                val_dloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                val_dloaders.append(val_dloader)
                batch_sizes.append(batch_size)
                adv_transforms.append(adv_transform)
                loss_fns.append(loss_fn)
                cps.append(cp)
                weights.append(weight)
                p_dropouts.append(p_dropout)
                excl_from_dropouts.append(excl_from_dropout)
            train_dloader = MultiDataLoader(train_sets, batch_sizes=batch_sizes, num_iterations=self.config["steps_per_epoch"])
            n_epochs = self.config["epochs"]
        else:
            learning_rate = self.config["train_args"]["learning_rate"]
            train_set, val_set, batch_size, adv_transform, excl_from_dropout, loss_fn, cp = config_to_trainset_valset_batchsize_advtransform(self.config)
            train_dloader = MultiDataLoader([train_set], batch_sizes=[batch_size], num_iterations=len(train_set)//batch_size)
            val_dloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            val_dloaders = [val_dloader]
            adv_transforms = [adv_transform]
            excl_from_dropouts = [excl_from_dropout]
            loss_fns = [loss_fn]
            n_epochs = self.config["train_args"]["epochs"]
            cps = [cp]
            weights = [1.0]
            
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # Training
        def loss_from_data(loss_fn, data, adv_transform, excl_from_dropout, p_dropout):
            if adv_transform is not None:
                adv_data = adv_transform(data.clone())
                adv_data = adv_data.to(self.config["device"])
                data = data.to(self.config["device"])
                if excl_from_dropout:
                    exclude_from_dropout = adv_data.adv_affected_nodes
                else:
                    exclude_from_dropout = None
                y_adv, dropout_mask = model(adv_data, exclude_from_dropout=exclude_from_dropout, return_dropout_mask=True, p_dropout=p_dropout)
                y_pred = model(data, dropout_mask=dropout_mask, p_dropout=p_dropout)
                loss = loss_fn(y_pred, y_adv, data.y, model.logit_scale)
            else:
                data = data.to(self.config["device"])
                loss = loss_fn(model(data, p_dropout=p_dropout), data.y, model.logit_scale)
            return loss

        pbar_epochs = tqdm(range(n_epochs), position=0)
        smallest_val_losses = np.inf
        for epoch in pbar_epochs:
            # Train
            model.train()
            train_losses = []
            mov_avg_train_losses = 0
            pbar_train = tqdm(train_dloader, position=1, leave=False)
            for datas in pbar_train:
                optimizer.zero_grad()
                losses = []
                total_loss = 0
                for loss_fn, data, adv_transform, excl_from_dropout, weight, p_dropout in zip(loss_fns, datas, adv_transforms, excl_from_dropouts, weights, p_dropouts):
                    loss = loss_from_data(loss_fn, data, adv_transform, excl_from_dropout, p_dropout)
                    total_loss += weight * loss
                    losses.append(loss)
                total_loss.backward()
                optimizer.step()
                train_losses.append([l.item() for l in losses])
                mov_avg_train_losses = 0.9 * mov_avg_train_losses + 0.1 * torch.tensor(train_losses[-1])
                pbar_train.set_postfix({'moving average train_loss': mov_avg_train_losses.tolist()})
            train_losses_mean = np.array(train_losses).mean(axis=0)
            losses = []
            # Validate
            model.eval()
            val_losses_mean = []
            for val_dloader, loss_fn, adv_transform in zip(val_dloaders, loss_fns, adv_transforms):
                val_losses = []
                mov_avg_val_loss = 0
                pbar_val = tqdm(val_dloader, position=1, leave=False)
                for data in pbar_val:
                    with torch.no_grad():
                        loss = loss_from_data(loss_fn, data, adv_transform, False, 0.0)
                        val_losses.append(loss.item())
                    mov_avg_val_loss = 0.9 * mov_avg_val_loss + 0.1 * loss.item()
                    pbar_train.set_postfix({'moving average val_loss': mov_avg_val_loss})
                val_losses_mean.append(np.mean(val_losses))
            # Log
            def loss2str(losses):
                return np.array2string(np.array(losses), precision=3)
            pbar_epochs.set_postfix({'train_losses': loss2str(train_losses_mean), 'val_loss': loss2str(val_losses_mean)})
            logging.info(f"Epoch {epoch}, average train_loss: {loss2str(train_losses_mean)}, average val_loss: {loss2str(val_losses_mean)}")
            # Save Checkpoint
            cps = np.array(cps)
            if np.any(np.array(val_losses_mean) < np.minimum(smallest_val_losses, cps)):
                do_cp = True
                smallest_val_losses = np.minimum(val_losses_mean, smallest_val_losses)
            else:
                do_cp = False
            if do_cp:
                logging.info(f"Saving checkpoint...")
                model.cpu()
                torch.save(model, osp.join(self.config["experiment_dir"], f"checkpoint_{epoch+1}_{loss2str(val_losses_mean)}.pt"))
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
        elif self.config["dataset"] == "VisualGenomeAdversarial2":
            dataset = VisualGenomeAdversarial2(**self.config["dataset_args"])
            dataset = dataset_filter(dataset, **self.config["dataset_filter_args"])
            train_val_split = self.config["eval_args"]["train_val_split"]
            train_ratio = train_val_split
            _, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
            val_set = dataset_filter(val_set, **self.config["valset_filter_args"])
            val_dloader = DataLoader(val_set, batch_size=1, shuffle=False)
        elif self.config["dataset"] == "VisualGenomeAdversarialAttr":
            dataset = VisualGenomeAdversarialAttr(**self.config["dataset_args"])
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