import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class InputModule(nn.Module):
    def __init__(self, num_poiType):
        super(InputModule, self).__init__()
        self.locS_map = nn.Linear(num_poiType, 3, bias=False)
        self.day_week_emb = nn.Embedding(7, 3)
        self.tS_depart_emb = nn.Embedding(24 * 2, 6)
        self.locS_point_emb = nn.Embedding(num_poiType + 1, 3, padding_idx=0)

    @property
    def output_size(self):
        """5 == len(INPUT_FLOAT)"""
        return self.locS_map.out_features + self.day_week_emb.embedding_dim + \
            self.tS_depart_emb.embedding_dim + self.locS_point_emb.embedding_dim + 5

    @property
    def s_depart_size(self):
        return self.locS_map.out_features + self.day_week_emb.embedding_dim + \
               self.tS_depart_emb.embedding_dim

    def forward(self, meta_int, meta_float, seq_int, seq_float):
        locS_depart = self.locS_map(meta_float.squeeze(-1))
        day_week, tS_depart = meta_int.unbind(dim=-1)
        day_week = self.day_week_emb(day_week)
        tS_depart = self.tS_depart_emb(tS_depart)
        s_depart = torch.cat([locS_depart, day_week, tS_depart], dim=-1)
        locS_point = self.locS_point_emb(seq_int.squeeze(-1) + 1)
        seq = torch.cat([s_depart.unsqueeze(0).expand(seq_float.size(0), -1, -1),
                         locS_point, seq_float], dim=-1)
        score = self.loc_importance(seq_float[:, :, -3:])
        return s_depart, seq, score

    @classmethod
    def loc_importance(cls, ds):
        speed, dis, angle = ds.unbind(dim=-1)
        speed = 1.0 / (speed + 1.0)
        return torch.stack([speed, dis, angle], dim=-1)


class TravelPatternsModule(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_layer=2, dropout=0.1):
        super(TravelPatternsModule, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layer)
        self.dropout1 = nn.Dropout(dropout)
        self.spatial_attn = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Softmax(dim=0)
        )

    def forward(self, sequence, locI, mask):
        mask_inv = ~mask
        length = torch.count_nonzero(mask_inv, dim=1).cpu()
        x = pack_padded_sequence(sequence, length.cpu(), enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, total_length=sequence.size(0))[0]
        x = self.dropout1(x)
        locI = self.spatial_attn(locI) * mask_inv.mT.unsqueeze(-1).to(torch.float)
        locI = locI / locI.sum(dim=0, keepdims=True)
        x = (x * locI).sum(dim=0)
        return x


class ResidualStructure(nn.Module):
    def __init__(self, input_size, hidden_size, n_block=4, dropout=0.1):
        super(ResidualStructure, self).__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        residuals = []
        for i in range(n_block):
            residuals.append(nn.Linear(hidden_size, hidden_size))
        self.residuals = nn.ModuleList(residuals)
        self.act = nn.LeakyReLU()
        self.final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, s_depart, traj_pattern):
        x = torch.cat([s_depart, traj_pattern], dim=-1)
        x = self.act(self.input2hid(x))
        for layer in self.residuals:
            x = self.act(layer(x)) + x
        return self.final(x)


class LSI_LSTM(nn.Module):
    def __init__(self, num_poiType, hidden_size=128, n_lstm_layer=2,
                 n_res_block=4, dropout=0.1):
        super(LSI_LSTM, self).__init__()
        self.input_module = InputModule(num_poiType)
        self.travel_pattern = TravelPatternsModule(
            self.input_module.output_size, hidden_size, n_lstm_layer, dropout)
        self.residual_structure = ResidualStructure(
            self.input_module.s_depart_size + hidden_size, hidden_size,
            n_res_block, dropout)

    def forward(self, meta_int, meta_float, seq_int, seq_float, seq_mask):
        s_depart, seq, score = self.input_module(
            meta_int, meta_float, seq_int, seq_float)
        hidden = self.travel_pattern(seq, score, seq_mask)
        result = self.residual_structure(s_depart, hidden)
        return result
