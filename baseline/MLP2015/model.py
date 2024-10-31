import torch
from torch import nn
from torch.nn.functional import softmax


class MLP(nn.Module):
    """
    Kaggle 2015比赛的胜出模型，结构详见De Brébisson et al., (2015)

    :param centroids: 聚类中心
    :param split_size: 截取轨迹的出发前k个点和当前位置前k个点
    """
    def __init__(self, centroids, emb_dim=10, hidden_dim=500, split_size=5, dropout=0.1):
        super(MLP, self).__init__()
        self.split_size = split_size
        self.week_emb = nn.Embedding(5, emb_dim)
        self.weekday_emb = nn.Embedding(7, emb_dim)
        self.time_emb = nn.Embedding(24 * 4, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(split_size * 4 + emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, centroids.size(0)),
            nn.Softmax(dim=-1)
        )
        centroid_layer = nn.Linear(centroids.size(0), centroids.size(1), bias=False)
        centroid_layer.weight = nn.Parameter(centroids.mT)
        centroid_layer.requires_grad_(False)
        self.centroid = centroid_layer

    def forward(self, meta_int, meta_float, seq_int, seq_float, seq_mask):
        x = self.emb_input(meta_int, seq_float, seq_mask)
        x = self.mlp(x)
        return self.centroid(x)

    def emb_input(self, meta, seq, mask):
        week, weekday, hour = meta.unbind(dim=-1)
        week = self.week_emb(week)
        weekday = self.weekday_emb(weekday)
        hour = self.time_emb(hour)
        start = seq[:self.split_size, :, :].transpose(0, 1).flatten(1, -1)
        final_idx = torch.count_nonzero(~mask, dim=1) - 1
        final_idx = final_idx.unsqueeze(0).unsqueeze(-1) \
            .expand(self.split_size, seq.size(1), seq.size(2))
        index = torch.arange(1 - self.split_size, 1, device=final_idx.device).unsqueeze(-1).unsqueeze(-1)
        final_idx = final_idx + index
        end = seq.gather(0, final_idx).transpose(0, 1).flatten(1, -1)
        return torch.cat([start, end, week, weekday, hour], dim=-1)
