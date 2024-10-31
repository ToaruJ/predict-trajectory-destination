import numpy as np
import torch
from torch import nn


class TES(nn.Module):
    def __init__(self, num_grid_axis, len_context, d_model=128, n_head=8,
                 n_layer=2, dropout=0.05, max_len=256):
        super(TES, self).__init__()
        self.grid_emb = nn.Embedding(num_grid_axis * num_grid_axis + 1, d_model, padding_idx=0)
        self.register_buffer('pe', self.init_positional_encode(max_len, d_model))
        self.context_emb = nn.Embedding(24 * 4, d_model)
        self.cat_linear = nn.Linear(d_model * 3, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder, n_layer)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2))

    @classmethod
    def init_positional_encode(cls, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, meta_int, meta_float, seq_int, seq_float, seq_mask,
                output_seq=False, predict_final=False):
        emb = self.grid_emb(seq_int.squeeze(-1) + 1)
        pos_encode = self.positional_encode(emb)
        context = self.context_emb(meta_int.squeeze(-1)).expand(emb.size(0), -1, -1)
        seq = torch.concat([emb, pos_encode, context], dim=-1)
        x = self.cat_linear(seq)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(emb.size(0)).to(emb.device)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=seq_mask)
        x = x.transpose(0, 1)
        if predict_final:
            final_idx = torch.count_nonzero(~seq_mask, dim=1) - 1
            final_idx = final_idx.unsqueeze(-1).unsqueeze(-1)
            final_idx = final_idx.expand(x.size(0), 1, x.size(2))
            x = torch.gather(x, 1, final_idx).squeeze(1)
        x = self.mlp(x)
        if output_seq:
            return x, None, context, seq
        return x, None

    def positional_encode(self, sequence):
        return self.pe[:sequence.size(0)].expand(-1, sequence.size(1), -1)
