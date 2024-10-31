import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class TNV(nn.Module):
    """
    skip-gram model for neighbouring dependencies
    """
    def __init__(self, num_roads, emb_dim=32):
        super(TNV, self).__init__()
        self.num_roads = num_roads
        self.emb_dim = emb_dim
        self.center_emb = nn.Embedding(num_roads, emb_dim)
        self.context_emb = nn.Linear(emb_dim, num_roads)

    def forward(self, x):
        return self.context_emb(self.center_emb(x))

    @torch.no_grad()
    def embed(self, x):
        return self.center_emb(x)

    def embedding_center(self):
        return self.center_emb.weight.detach()

    def embedding_context(self):
        return self.context_emb.weight.detach().mT


class DND(nn.Module):
    def __init__(self, tnv_emb, centroids, emb_dim=32, dropout=0.1):
        super(DND, self).__init__()
        _weight = torch.cat([torch.zeros(1, tnv_emb.size(1), dtype=tnv_emb.dtype),
                             tnv_emb], dim=0)
        self.tnv_emb = nn.Embedding.from_pretrained(_weight, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.tnv_emb.embedding_dim, hidden_size=emb_dim,
                            dropout=dropout)
        self.fcn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, centroids.size(0)),
            nn.Softmax(dim=-1)
        )
        centroid_layer = nn.Linear(centroids.size(0), centroids.size(1), bias=False)
        centroid_layer.weight = nn.Parameter(centroids.mT)
        centroid_layer.requires_grad_(False)
        self.centroid = centroid_layer

    def forward(self, meta_int, meta_float, seq_int, seq_float, seq_mask,
                predict_final=False, output_seq=False):
        emb = self.tnv_emb(seq_int.squeeze(-1) + 1)
        length = torch.count_nonzero(~seq_mask, dim=1)
        x = pack_padded_sequence(emb, length.cpu(), enforce_sorted=False)
        output, (h, _) = self.lstm(x)
        if predict_final:
            x = h.squeeze(0)
        else:
            x = pad_packed_sequence(output, total_length=seq_int.size(0))[0]
            x = x.transpose(0, 1)
        x = self.centroid(self.fcn(x))
        if output_seq:
            # return target_road, trip_time, sequence[1:, ...], new_mask[:, 1:]
            return x, None, None, emb
        return x, None
