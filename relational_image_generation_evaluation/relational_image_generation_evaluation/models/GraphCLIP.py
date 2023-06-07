import torch
import torch.nn.functional as F

from .model_utils import global_master_pool, tokens_to_embeddings_batched
        
# Like GNN15, but adds input noise and layer dropout.
class GNN15(torch.nn.Module):
    def forward(self, data):
        data = tokens_to_embeddings_batched(data, self.embedding)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.project_edges(edge_attr)
        x, edge_attr, _ = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x, edge_attr, _ = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x, edge_attr, _ = self.conv3(x, edge_index, edge_attr)
        x = global_master_pool(x, batch)
        return x
