# 轨迹目的地预测模型：主要模型的框架部分


import numpy as np
from sklearn.neighbors import KernelDensity
import torch
from torch import nn
from globalval import *


__all__ = ['Time2Vec', 'EmbeddingLayer', 'PositionalEncoding',
           'TransfmPredictor', 'DiffusionPredictor']


class Time2Vec(nn.Module):
    """
    时间的嵌入向量。详见Kazemi et al., 2019。
    time2vec的定义：t2v(t)_i = F(w_i * t + phi_i),
        F(x) = x if i == 0 else F(x) = sin(x)
    最终实验效果表明，直接用离散的24h（0-23值）进行嵌入，比time2vec效果好，因而弃用
    """
    def __init__(self, emb_dim):
        super(Time2Vec, self).__init__()
        self.layer = nn.Linear(1, emb_dim)

    def forward(self, t):
        t = self.layer(t)
        return torch.cat([t[..., 0:1], torch.sin(t[..., 1:])], dim=-1)


class ResidualBlock(nn.Module):
    """2个线性层组成的短路block"""
    def __init__(self, emb_dim, dim_forward=None, output_ratio=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        dim_forward = emb_dim if dim_forward is None else dim_forward
        self.lin1 = nn.Linear(emb_dim, dim_forward)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.lin2 = nn.Linear(dim_forward, emb_dim * output_ratio)
        self.norm = nn.LayerNorm(emb_dim * output_ratio)
        self.output_ratio = output_ratio

    def forward(self, x):
        y = self.lin2(self.dropout(self.act(self.lin1(x))))
        if self.output_ratio > 1:
            y = self.norm(x.tile(self.output_ratio) + y)
            return y.chunk(self.output_ratio, dim=-1)
        else:
            return self.norm(x + y)


class EmbeddingLayer(nn.Module):
    """
    将轨迹点的记录信息、轨迹属性嵌入到模型中。

    :param emb_dim: 嵌入向量的维度
    :param road_num: 路段的数量，当不使用预训练路段向量时使用
    :param road_weight: 已经训练好的路段向量
    :param poi_weight: 已经训练好的POI特征向量
    """
    def __init__(self, emb_dim, road_num=None, road_weight=None, poi_weight=None):
        super(EmbeddingLayer, self).__init__()
        self.emb_dim = emb_dim
        assert road_num is not None or road_weight is not None

        # 对轨迹序列的路段特征（整数index）进行embed
        if road_weight is None:
            self.road1_id_emb = nn.Embedding(road_num + 1, emb_dim, padding_idx=0)
        else:
            _weight = torch.cat([torch.zeros(1, road_weight.size(1), dtype=road_weight.dtype),
                                 road_weight], dim=0)
            _road1_id_emb = nn.Embedding.from_pretrained(_weight, freeze=True, padding_idx=0)
            self.road1_id_emb = nn.Sequential(_road1_id_emb,
                                              nn.Linear(_weight.size(1), emb_dim))
        self.road_class_emb = nn.Embedding(len(ROADCLASS_map) + 1, emb_dim, padding_idx=0)
        self.direction_emb = nn.Embedding(5, emb_dim, padding_idx=0)
        self.form_way_emb = nn.Embedding(len(FORMWAY_map) + 1, emb_dim, padding_idx=0)
        self.link_type_emb = nn.Embedding(len(LINKTYPE_map) + 1, emb_dim, padding_idx=0)
        self.traj_emb = nn.Linear(len(INPUT_FLOAT) - 1 - len(POI_dict), emb_dim)
        self.tm_emb = Time2Vec(emb_dim)

        # 对路段周围的POI信息、轨迹元数据进行embed
        if poi_weight is None:
            self.poi_emb = nn.Linear(len(POI_dict), emb_dim)
            self.poi_emb_meta = nn.Linear(len(POI_dict), emb_dim)
        else:
            _poi_emb = nn.Linear(poi_weight.size(0), poi_weight.size(1))
            _poi_emb.weight = nn.Parameter(poi_weight.mT)
            self.poi_emb = nn.Sequential(_poi_emb, nn.Linear(poi_weight.size(1), emb_dim))
            self.poi_emb_meta = nn.Sequential(_poi_emb, nn.Linear(poi_weight.size(1), emb_dim))
        self.weekday_emb = nn.Embedding(7, emb_dim)
        self.is_vacation_emb = nn.Embedding(2, emb_dim)
        self.time_emb = nn.Embedding(24 * 4, emb_dim)
        self.meta_f_emb = nn.Linear(len(INPUT_META_FLOAT) - len(POI_dict), emb_dim)

    def forward(self, meta_int, meta_float, seq_int, seq_float):
        """
        :param meta_int: 轨迹元数据：离散整数部分
        :param meta_float: 轨迹元数据：浮点数连续值部分
        :param seq_int: 轨迹序列：离散整数部分
        :param seq_float: 轨迹序列：浮点数连续值部分
        :return: (轨迹元数据的嵌入, 轨迹序列的嵌入) ->
                 shape = ((Length1, b_size, emb_dim), (Length2, b_size, emb_dim))
        """
        # 目前的拼接方式是：元数据之间考虑attention，同一个轨迹点的属性直接相加
        seq_emb = torch.stack(self._forward_seq_int(seq_int) + self._forward_seq_float(seq_float)) \
            .mean(dim=0)
        return self._forward_meta_emb(meta_int, meta_float), seq_emb

    def _forward_meta_emb(self, meta_int, meta_float):
        """轨迹元数据处理"""
        weekday, is_vacation, hour = meta_int.unbind(dim=-1)
        meta_f, pois = meta_float.split([len(INPUT_META_FLOAT) - len(POI_dict), len(POI_dict)],
                                        dim=-1)
        return torch.stack([
            self.weekday_emb(weekday),
            self.is_vacation_emb(is_vacation),
            self.time_emb(hour),
            self.meta_f_emb(meta_f),
            self.poi_emb_meta(pois)
        ], dim=0)

    def _forward_seq_int(self, seq_int):
        """seq_int处理"""
        seq_int = seq_int + 1
        road1_id, road_class, direction, form_way, link_type = seq_int.unbind(dim=-1)
        return [self.road1_id_emb(road1_id), self.road_class_emb(road_class),
                self.direction_emb(direction), self.form_way_emb(form_way),
                self.link_type_emb(link_type)]

    def _forward_seq_float(self, seq_float):
        """seq_float处理"""
        traj_floats, tm, pois = seq_float.split([len(INPUT_FLOAT) - 1 - len(POI_dict),
                                                 1, len(POI_dict)], dim=-1)
        return [self.traj_emb(traj_floats), self.tm_emb(tm), self.poi_emb(pois)]


class PositionalEncoding(nn.Module):
    """
    位置编码模块，遵循sin和cos交替的位置编码。
    PosEncoder(pos, 2i) = sin(pos/10000^(2i/d_model))
    PosEncoder(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    其中pos是在序列中的位置、i是特征维度的下标

    :param d_model: 特征向量的维度
    :param batch_first: True表示输入数据格式为(bsize, seq_len, d)，False表示(seq_len, bsize, d)
    """
    def __init__(self, d_model, max_len=256, batch_first=False):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        self.batch_first = batch_first
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(0), :] if self.batch_first \
            else x + self.pe[:x.size(0)]

    def get_position(self, pos):
        return self.pe[0, pos, :] if self.batch_first else self.pe[pos, 0, :]


class TransformerWithMeta(nn.Module):
    """拼接序列信息、元数据信息的Transformer encoder"""
    def __init__(self, n_layer, emb_dim, n_head, dim_feedforward=None, dropout=0.1, activation='relu'):
        super(TransformerWithMeta, self).__init__()
        dim_feedforward = emb_dim if dim_feedforward is None else dim_feedforward
        self.n_layer = n_layer
        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(
            emb_dim, n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
            for _ in range(n_layer)])
        self.meta_proj = ResidualBlock(emb_dim, dim_feedforward, output_ratio=n_layer, dropout=dropout)

    def forward(self, meta, seq, mask=None, src_key_padding_mask=None, output_feat=False):
        """
        :param meta: size = (len_meta, batch size, emb_dim)
        """
        metas = self.meta_proj(meta)
        meta_size = metas[0].size(0)
        if mask is not None:
            mask = torch.cat([torch.ones(meta_size, mask.size(1), dtype=torch.bool, device=mask.device),
                              mask], dim=0)
            mask = torch.cat([torch.zeros(mask.size(0), meta_size, dtype=torch.bool, device=mask.device),
                              mask], dim=1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat([
                torch.zeros(src_key_padding_mask.size(0), meta_size, dtype=torch.bool,
                            device=src_key_padding_mask.device),
                src_key_padding_mask], dim=1)
        inner_feat = []
        for layer, meta in zip(self.encoders, metas):
            seq = torch.cat([meta, seq], dim=0)
            seq = layer(seq, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            _, seq = seq.split([meta_size, seq.size(0) - meta_size], dim=0)
            if output_feat:
                inner_feat.append(seq)
        if output_feat:
            return seq, torch.cat(inner_feat, dim=-1)
        return seq


class AttentionDecoderLayer(nn.Module):
    """
    类似于Transformer DecoderLayer，但是去掉了self attention，
    只有multihead attention + feedforward
    """
    def __init__(self, emb_dim, n_head, dim_feedforward=None, kv_dim=None, dropout=0.1):
        super(AttentionDecoderLayer, self).__init__()
        dim_feedforward = emb_dim if dim_feedforward is None else dim_feedforward
        self.multihead_attn = nn.MultiheadAttention(
            emb_dim, n_head, dropout=dropout, kdim=kv_dim, vdim=kv_dim)
        self.linear1 = nn.Linear(emb_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        x = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask,
                                need_weights=False)[0]
        x = self.dropout2(x)
        tgt = self.norm2(tgt + x)
        x = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        x = self.dropout3(x)
        tgt = self.norm3(tgt + x)
        return tgt


class AttentionDecoder(nn.Module):
    """多层Attention Decoder Layer的组合"""
    def __init__(self, n_layer, emb_dim, n_head, dim_feedforward=None, kv_dim=None, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        self.decoder = nn.ModuleList([
            AttentionDecoderLayer(emb_dim, n_head, dim_feedforward, kv_dim, dropout)
            for _ in range(n_layer)])

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        for layer in self.decoder:
            tgt = layer(tgt, memory, memory_mask=memory_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        return tgt


class TransfmPredictor(nn.Module):
    """
    使用Transformer预测轨迹目的地。以Transformer with meta为backbone的预测模型。
    该模型将轨迹元数据emb、序列记录点的embed在序列时间维度拼接起来，作为key, value
    序列部分使用time2vec进行序列关系建模。
    对Transformer的输出序列，经过decoder得到每步的预测结果。
    注：默认batch_first = False

    :param emb_dim: 隐藏层的维度，主要的隐藏层都是这个维度，
                    多层Linear之间的`dim_feedforward` = `emb_dim` * 4
    :param road_num: 路段的数量，当不使用预训练路段向量时使用
    :param road_weight: 已经训练好的路段向量，shape = (num_road, emb_dim)
    :param poi_weight: 已经训练好的POI特征向量，shape = (num_POItype, emb_dim)
    :param decode_roademb: 输入的路段向量`road_weight`同时作为decoder的最后一层参数
    """
    def __init__(self, emb_dim, n_layer=4, n_head=4, activation='relu', max_len=256, dropout=0.1,
                 road_num=None, road_weight=None, poi_weight=None, decode_roademb=False,
                 y_num=1):
        super(TransfmPredictor, self).__init__()
        assert (not decode_roademb) or road_weight is not None

        self.emb_dim = emb_dim
        self.decode_roademb = decode_roademb
        self.embed = EmbeddingLayer(emb_dim, road_num, road_weight, poi_weight)
        # encoder_layer = nn.TransformerEncoderLayer(emb_dim, n_head, dim_feedforward=emb_dim * 4,
        #                                            dropout=dropout, activation=activation)
        # self.start_emb = nn.Embedding(1, emb_dim)
        self.transformer = TransformerWithMeta(n_layer, emb_dim, n_head, dim_feedforward=emb_dim * 4,
                                               dropout=dropout, activation=activation)
        self.decoder1 = nn.Sequential(nn.Linear(emb_dim, emb_dim * 4),
                                      nn.LeakyReLU(),
                                      nn.Linear(emb_dim * 4, emb_dim),
                                      nn.Dropout(dropout))
        self.decoder_float = nn.Linear(emb_dim, len(INPUT_Y_FLOAT) * y_num)
        self.y_num = y_num
        # if decode_roademb:
        #     _decoder_road = nn.Linear(road_weight.size(1), road_weight.size(0), bias=False)
        #     _decoder_road.weight = nn.Parameter(road_weight)
        #     _decoder_road.requires_grad_(False)
        #     self.decoder_road = nn.Sequential(nn.Linear(emb_dim, road_weight.size(1)),
        #                                       _decoder_road)
        # else:
        #     _final_dim = road_num if road_num is not None else road_weight.size(0)
        #     self.decoder_road = nn.Linear(emb_dim, _final_dim)

    def forward(self, meta_int, meta_float, seq_int, seq_float, seq_mask,
                output_seq=False, predict_final=False, pred_y_index=None):
        """
        :param meta_int: 轨迹元数据：离散整数部分
        :param meta_float: 轨迹元数据：浮点数连续值部分
        :param seq_int: 轨迹序列：离散整数部分
        :param seq_float: 轨迹序列：浮点数连续值部分
        :param seq_mask: 轨迹序列的mask（boolTensor, True表示pad值）
        :param output_seq: True表示同时输出最后一层Transformer的序列结果和mask，False表示只输出目的地预测值
        :param predict_final: True表示仅输出每条序列最后一个时间步的预测结果（用于验证、测试）
        :param pred_y_index: 输出的x, y下标（即第i个损失函数训练后的输出），None表示全输出
        :return: 目的地路段的预测logit, 预测旅途用时, 后一层Transformer的输出, 输出对应的mask
        """
        meta, sequence = self.embed(meta_int, meta_float, seq_int, seq_float)
        attn_mask = self._attention_mask(sequence, 0)
        result = self.transformer(
            meta, sequence, mask=attn_mask, src_key_padding_mask=seq_mask, output_feat=output_seq)
        inner_feat = None
        if output_seq:
            result, inner_feat = result
        result = result.transpose(0, 1)
        if predict_final:
            final_idx = torch.count_nonzero(~seq_mask, dim=1) - 1
            final_idx = final_idx.unsqueeze(-1).unsqueeze(-1)
            final_idx = final_idx.expand(result.size(0), 1, result.size(2))
            result = torch.gather(result, 1, final_idx).squeeze(1)
        result = self.decoder1(result) + result
        # target_road = self.decoder_road(result)
        dest_xy, trip_time = self.decoder_float(result).split([
            2 * self.y_num, (len(INPUT_Y_FLOAT) - 2) * self.y_num], dim=-1)
        if pred_y_index is not None:
            dest_xy = dest_xy.chunk(self.y_num, dim=-1)[pred_y_index]
            trip_time = trip_time.chunk(self.y_num, dim=-1)[pred_y_index]
        if output_seq:
            # return target_road, trip_time, sequence[1:, ...], new_mask[:, 1:]
            return dest_xy, trip_time, meta, inner_feat
        return dest_xy, trip_time

    @classmethod
    def _attention_mask(cls, total_seq, meta_size):
        sz = total_seq.size(0)
        mask = torch.triu(torch.full((sz, sz - meta_size), 1, dtype=torch.bool, device=total_seq.device),
                          diagonal=1-meta_size)
        if meta_size != 0:
            mask = torch.cat([torch.zeros(sz, meta_size, dtype=torch.bool, device=mask.device),
                              mask], dim=1)
        return mask


class DiffusionPredictor(nn.Module):
    """
    使用Transformer + DDPM预测轨迹目的地。以Transformer Encoder为序列建模backbone，
    Decoder为扩散噪声预测部分，可以认为是DDPM的e_theta(x, y_t, f(x)=E(y|x), t)。
    参见CARD(Classification And Regression Diffusion)模型(Han et al., 2022)

    :param input_dim: 变量y的维度：若是分类问题，是道路路段的数量（类别数）；回归问题，是待预测xy的维度
    :param encoder: 预训练好的编码器，可以给出E(y|x)的初步预测值
    :param emb_dim: 隐藏层的维度，主要的隐藏层都是这个维度，
                    多层Linear之间的`dim_feedforward` = `emb_dim` * 4
    :param diff_steps: 扩散过程的步数
    :param schedule: 扩散强度的函数，'cosine', 'linear'
    """
    def __init__(self, input_dim, encoder: TransfmPredictor, emb_dim, diff_steps=100,
                 schedule='linear', n_layer=4, n_head=4, activation='relu', dropout=0.1):
        assert schedule in ['cosine', 'linear']
        super(DiffusionPredictor, self).__init__()

        self.input_dim = input_dim
        self.diff_steps = diff_steps
        encoder.requires_grad_(False)
        encoder.eval()
        self.encoder = encoder
        self.step_emb = nn.Embedding(diff_steps, emb_dim)

        self.y_layer = nn.Sequential(nn.Linear(input_dim * 2, emb_dim * 4),
                                     nn.LeakyReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(emb_dim * 4, emb_dim))
        self.y_time_layer = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4),
                                          nn.LeakyReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(emb_dim * 4, emb_dim))

        self.diffusion = AttentionDecoder(n_layer, emb_dim, n_head, dim_feedforward=emb_dim * 4,
                                          kv_dim=encoder.transformer.n_layer * emb_dim,
                                          dropout=dropout)
        self.final_decode = nn.Sequential(nn.Linear(emb_dim, emb_dim * 4),
                                          nn.LeakyReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(emb_dim * 4, input_dim))

        # 扩散过程的系数
        if schedule == 'cosine':
            alphas = self.cosine_schedule(diff_steps)
        elif schedule == 'linear':
            alphas = self.linear_schedule(diff_steps)
        else:
            raise NotImplementedError
        alpha_bar, sqrt_alpha_bar, gamma0, gamma1, gamma2, sqrt_beta_tilde = \
            self.cal_schedule_parameters(alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('gamma0', gamma0)
        self.register_buffer('gamma1', gamma1)
        self.register_buffer('gamma2', gamma2)
        self.register_buffer('sqrt_beta_tilde', sqrt_beta_tilde)

    def forward(self, x_meta, x_seq, seq_mask, y_pred, noised_y, t):
        """
        :param x_meta: 条件扩散 - 限定的输入条件：元数据信息
        :param x_seq: 条件扩散 - 限定的输入条件：序列信息
        :param y_pred: encoder预测的y期望值
        :param noised_y: 添加了噪声后的预测结果
        :param t: 噪声强度水平（添加噪声的步数），t in [1, self.diff_steps]
        :return: 预测值：添加的噪声值
        """
        if noised_y.ndim < x_seq.ndim:
            noised_y = noised_y.unsqueeze(0)
        y_pred = y_pred.expand_as(noised_y)
        y = torch.cat([noised_y, y_pred], dim=-1)
        y = self.y_layer(y)
        step_emb = self.step_emb(t - 1)
        y = torch.cat([y, step_emb.expand_as(y)], dim=-1)
        y = self.y_time_layer(y)
        attn_mask = None if y.size(0) == 1 else \
            nn.Transformer.generate_square_subsequent_mask(x_seq.size(0)).to(y.device)
        y = self.diffusion(y, x_seq, memory_mask=attn_mask,
                           memory_key_padding_mask=seq_mask)
        y = self.final_decode(y)
        return y


    @classmethod
    def cosine_schedule(cls, timesteps, s=0.008):
        """
        按照cosine schedule输出alpha。
        alpha_bar_t = f_t / f_0,
        f_t = cos(((t / T) + s) / (1 + s) * pi / 2) ^ 2
        """
        x = torch.arange(timesteps + 1, dtype=torch.float)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(alphas, min=0.001, max=1.)

    @classmethod
    def linear_schedule(cls, timesteps):
        """
        按照线性schedule输出alpha。
        beta_t = a + b * t, beta_1 = 1e-4, beta_N = 0.1, alpha_t = 1 - beta_t
        """
        beta = torch.linspace(1e-4, 0.1, timesteps)
        return 1 - beta

    @classmethod
    def cal_schedule_parameters(cls, alphas: torch.Tensor):
        """
        根据给定的alphas参数，计算扩散过程需要的系数常数

        :return: alpha_bar, sqrt_alpha_bar, gamma_0, gamma_1, gamma_2, sqrt_beta_tilde
        """
        sqrt_alpha = alphas.sqrt()
        alpha_bar = alphas.cumprod(-1)
        sqrt_alpha_bar = alpha_bar.sqrt()
        beta = 1 - alphas
        alpha_tm1_bar = torch.cat([torch.ones(1, dtype=torch.float, device=alphas.device),
                                   alpha_bar[:-1]], dim=0)
        sqrt_alpha_tm1_bar = alpha_tm1_bar.sqrt()
        gamma0 = beta * sqrt_alpha_tm1_bar / (1 - alpha_bar)
        gamma1 = (1 - alpha_tm1_bar) * sqrt_alpha / (1 - alpha_bar)
        gamma2 = 1 + (sqrt_alpha_bar - 1) * (sqrt_alpha + sqrt_alpha_tm1_bar) / (1 - alpha_bar)
        beta_tilde = (1 - alpha_tm1_bar) / (1 - alpha_bar) * beta
        return alpha_bar, sqrt_alpha_bar, gamma0, gamma1, gamma2, beta_tilde.sqrt()

    def q_sample(self, y_0, y_pred, t, noise=None):
        """
        计算并采样q(y_t | y_0, y_pred)。
        y_t = sqrt(alpha_bar) * y_0 + (1 - sqrt(alpha_bar)) * y_pred
              + sqrt(1 - alpha_bar) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(y_0, device=y_0.device)
        alpha_bar = self.alpha_bar.gather(0, t - 1).unsqueeze(-1)
        sqrt_alpha_bar = self.sqrt_alpha_bar.gather(0, t - 1).unsqueeze(-1)
        return sqrt_alpha_bar * y_0 + (1 - sqrt_alpha_bar) * y_pred + \
               torch.sqrt(1 - alpha_bar) * noise

    def reparameter_y0_hat(self, y_t, y_pred, noise_pred, t):
        """
        计算y_0^hat = 1 / sqrt(alpha_bar) * (y_t - (1 - sqrt(alpha_bar)) * y_pred -
                                            sqrt(1 - alpha_bar) * noise_pred)
        """
        alpha_bar = self.alpha_bar.gather(0, t - 1).unsqueeze(-1)
        sqrt_alpha_bar = self.sqrt_alpha_bar.gather(0, t - 1).unsqueeze(-1)
        unnorm = y_t - (1 - sqrt_alpha_bar) * y_pred - torch.sqrt(1 - alpha_bar) * noise_pred
        return unnorm / sqrt_alpha_bar

    def p_sample(self, y_t, y_0, y_pred, t, noise=None):
        """
        计算并采样q(y_t-1 | y_t, y_0, y_pred)。
        y_t-1 = gamma_0 * y_0 + gamma_1 * y_t + gamma_2 + sqrt(beta_tilde) * z (if t > 1)
        """
        if noise is None:
            noise = torch.randn_like(y_0, device=y_0.device)
        gamma0 = self.gamma0.gather(0, t - 1).unsqueeze(-1)
        gamma1 = self.gamma1.gather(0, t - 1).unsqueeze(-1)
        gamma2 = self.gamma2.gather(0, t - 1).unsqueeze(-1)
        sqrt_beta_tilde = self.sqrt_beta_tilde.gather(0, t - 1).unsqueeze(-1)
        y_tm1 = gamma0 * y_0 + gamma1 * y_t + gamma2 * y_pred + sqrt_beta_tilde * noise
        t_broadcast = t.unsqueeze(-1).expand_as(y_0)
        return torch.where(t_broadcast <= 1, y_0, y_tm1)

    def p_sample_loop(self, x_meta, x_seq, seq_mask, y_pred):
        """
        从随机噪声y_T ~ N(f(x), I)开始，生成预测结果y_0
        """
        e = torch.randn(self.diff_steps, *y_pred.size(), device=y_pred.device)
        e = e.unbind(dim=0)
        y_t = y_pred + e[-1]
        for i in range(self.diff_steps, 0, -1):
            t = torch.full((y_pred.size(0),), i, device=y_t.device)
            pred_noise = self(x_meta, x_seq, seq_mask, y_pred, y_t, t)
            y0_reparam = self.reparameter_y0_hat(y_t, y_pred, pred_noise, t)
            y_t = self.p_sample(y_t, y0_reparam, y_pred, t, e[i - 2])
        return y_t.squeeze(0)

    @torch.inference_mode()
    def p_sample_loop_repeat(self, *x, repeat=32, n_per_batch=32, **kwargs):
        """
        多次采样，得到同一轨迹的多个预测结果，stack at dim=1

        :param x: 模型的输入，与self.encoder(*x)的输入相同，参见`TransfmPredictor`
        :param repeat: 采样次数
        :param n_per_batch: 在一个batch中的采样数，需要能被repeat整除。根据显存大小设定
        """
        assert repeat / n_per_batch == repeat // n_per_batch
        kwargs.setdefault('output_seq', True)
        kwargs.setdefault('predict_final', True)
        batch_samples = []
        cond_pred = self.encoder(*x, **kwargs)
        x_meta = cond_pred[2].tile(1, n_per_batch, 1)
        x_seq = cond_pred[3].tile(1, n_per_batch, 1)
        x_mask = x[4].tile(n_per_batch, 1)
        y_pred = cond_pred[0].tile(n_per_batch, 1)
        for _ in range(repeat // n_per_batch):
            data = self.p_sample_loop(x_meta, x_seq, x_mask, y_pred)
            batch_samples.extend(data.chunk(n_per_batch, dim=0))
        return torch.stack(batch_samples, dim=1)

    @classmethod
    def cal_dense_center(cls, pred):
        """
        从多次采样的结果，计算出进行指标运算的最终结果：取核密度最高的5个点，取均值

        :param pred: size = (batch, n_sample, y_dim)
        """
        result = []
        data = pred.detach().cpu().numpy()
        for i in range(data.shape[0]):
            _data = data[i]
            # 核密度估计
            std = np.sqrt(np.var(_data, axis=0).sum())
            kde = KernelDensity(bandwidth=std * 0.5)
            kde.fit(_data)
            score = kde.score_samples(_data)
            result.append(np.mean(_data[np.argsort(score)[-5:]], axis=0))
        result = np.stack(result, axis=0)
        return torch.as_tensor(result, device=pred.device)

    def train(self, mode=True):
        super(DiffusionPredictor, self).train(mode)
        self.encoder.eval()
        return self
