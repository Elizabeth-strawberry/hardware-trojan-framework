"""Independent fan-in/fan-out GraphSAGE encoders and supervised metric learning."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

from .data import FEATURE_NAMES


class GraphMentor(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        if (
            type(hidden_dim) is not int
            or hidden_dim < 1
            or type(num_layers) is not int
            or num_layers < 1
        ):
            raise ValueError("GraphMentor dimensions and layer count must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.config = {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout}
        self.fanin = nn.ModuleList()
        self.fanout = nn.ModuleList()
        for layer in range(num_layers):
            dim = len(FEATURE_NAMES) if layer == 0 else 2 * hidden_dim
            self.fanin.append(SAGEConv(dim, hidden_dim))
            self.fanout.append(SAGEConv(dim, hidden_dim))
        self.type_head = nn.Linear(2 * hidden_dim, 3)

    def forward(self, x, edge_index, batch=None):
        for fi, fo in zip(self.fanin, self.fanout):
            x = torch.cat((F.relu(fi(x, edge_index)), F.relu(fo(x, edge_index.flip(0)))), dim=-1)
            x = F.dropout(x, p=self.config["dropout"], training=self.training)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return F.normalize(x, dim=-1), F.normalize(global_mean_pool(x, batch), dim=-1)


def module_contrastive_loss(embeddings, module_labels, type_labels, margin=0.2):
    """Pull same-module nodes together and separate other modules within each type."""
    if not 0 <= margin < 1:
        raise ValueError("Cosine margin must be in [0, 1)")
    similarity = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).t()
    pairs = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    same_type = type_labels[:, None] == type_labels[None, :]
    same_module = module_labels[:, None] == module_labels[None, :]
    positive = pairs & same_type & same_module
    negative = pairs & same_type & ~same_module
    terms = []
    if positive.any():
        terms.append((1 - similarity[positive]).square().mean())
    if negative.any():
        terms.append(F.relu(similarity[negative] - margin).square().mean())
    return torch.stack(terms).mean() if terms else embeddings.sum() * 0
