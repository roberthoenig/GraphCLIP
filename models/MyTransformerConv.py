import math
from torch_geometric.nn import TransformerConv

import torch.nn.functional as F

from torch_geometric.utils import softmax


class MyTransformerConv(TransformerConv):
    def message(self, query_i, key_j, value_j,
                edge_attr, index, ptr,
                size_i):

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        # if edge_attr is not None:
            # out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out