# 生成基础地理数据的嵌入向量：路网嵌入

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.nn.pytorch as dglnn
import dgl.dataloading as dgldl
from pathlib import Path
from tqdm import tqdm
from globalval import TRAJ_ROOT, POI_dict
from preprocess_sz import geopoint_to_mp1


assert torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


class RoadNetworkDataLINE(Dataset):
    """
    数据集：路网的拓扑邻接数据，用于训练LINE (Tang et al. 2015)模型。
    输出的数据按照<center, positive, negatives>格式
    """
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)
        data['context'] = data.apply(lambda l: l['Fconnect'].split(' ')
                                               + l['Tconnect'].split(' '), axis=1)
        data['context'] = data['context'].apply(lambda item: list(map(int, filter(bool, item))))
        self.data = torch.tensor([[l.road1_id, item] for l in data.itertuples()
                                  for item in l.context], dtype=torch.long, device=DEVICE)
        self.num_roads = data['road1_id'].max() + 1
        freq = torch.bincount(self.data[:, 0])
        # 这里对出现频率q取 q^0.75，是遵循LINE原文中，负样本采样概率正比于 degree^0.75
        self.freq = freq ** 0.75
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item, 0], self.data[item, 1]

    def batch_negative_sample(self, batch_size, neg_sample):
        """
        负样本以batch形式直接获取，而不经过`__getitem__`方法和DataLoader。
        负样本的采样根据先验分布，而与正样本<center, positive>无关。
        因为性能问题，一次性取好单个batch的负样本，速度更快（反复调用multinomial很耗时）。
        对于加权的负样本采样，应该用multinomial方法
        """
        return torch.multinomial(self.freq.expand(batch_size, -1), neg_sample)


def create_graph_data_GAE(file_path, num_layer, neg_sample=5, batch_size=512):
    """
    读取文件，生成带有batch和负采样的graph和dataloader

    :param num_layer: 卷积层的数量n，采样数据时需要采样n跳的邻居
    :param neg_sample: 负采样数量
    :return: graph, dataloader
    """
    # 从文件中读取结点属性、图连边
    df = pd.read_csv(file_path)
    df.fillna({'Fconnect': '', 'Tconnect': ''}, inplace=True)
    fp = geopoint_to_mp1(df[['Flon', 'Flat']].values)
    tp = geopoint_to_mp1(df[['Tlon', 'Tlat']].values)
    df['connect'] = df['Fconnect'] + ' ' + df['Tconnect']

    def _str_to_ints(string):
        fil = filter(lambda s: s != '', string.split(' '))
        return list(map(int, fil))

    df['connect'] = df['connect'].apply(_str_to_ints)
    edges = [(line.road1_id, c) for line in df.itertuples() for c in line.connect]
    start, end = zip(*edges)

    # 载入到DGL.Graph中
    g = dgl.graph((torch.tensor(start), torch.tensor(end)),
                  num_nodes=df['road1_id'].max() + 1)
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g).to(DEVICE)
    g.ndata['feat'] = torch.tensor(np.concatenate([fp, tp], axis=1),
                                   dtype=torch.float, device=DEVICE)

    dataloader = dgldl.DataLoader(g, torch.arange(g.num_edges(), device=DEVICE),
        dgldl.as_edge_prediction_sampler(
            dgldl.MultiLayerFullNeighborSampler(num_layer),
            negative_sampler=dgldl.negative_sampler.Uniform(neg_sample)),
        device=DEVICE,
        batch_size=batch_size,
        shuffle=True)
    return g, dataloader


def _get_activation(act):
    """从文字到部分激活函数的映射"""
    def linear(x):
        return x

    if act is None:
        return linear
    if isinstance(act, str):
        act_map = {'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU, 'gelu': nn.GELU, 'silu': nn.SiLU,
                   'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        return act_map[act.lower()]()
    return act


class LINE(nn.Module):
    """
    LINE模型 (Tang et al. 2015)，通过一阶相似（直接连边）和二阶相似（相似的连边context），
    采用负采样方式（详见`RoadNetworkDataLINE`）训练结点的表示向量。
    分开训练等长度的<1st, 2nd> embedding，然后拼接成最终向量。
    训练时，每个路段对应一阶相似、二阶相似的center、context向量（3个），
    模型输出时，只取出、拼接<一阶相似, 二阶相似>两个向量

    :param num_road: 路段的数量
    :param emb_dim: 拼接后的表示向量的长度（其中前半部分是1st相似、
                    后半部分是2nd相似，两个向量等长）
    """
    def __init__(self, num_road, emb_dim=32):
        super(LINE, self).__init__()
        assert emb_dim % 2 == 0
        self.num_road = num_road
        self.emb_dim = emb_dim
        self.emb_1st = nn.Embedding(num_road, emb_dim // 2)
        self.emb_2nd = nn.Embedding(num_road, emb_dim // 2)
        self.emb_context = nn.Embedding(num_road, emb_dim // 2)

    def forward(self, center, pos, neg):
        """
        :return (pos1, pos2, neg1, neg2)的logit值。
            分别表示正样本的1阶相似性、2阶相似性、负样本的1阶相似性、2阶相似性
        """
        emb1 = self.emb_1st(center).unsqueeze(-2)
        emb2 = self.emb_2nd(center).unsqueeze(-2)
        embpos1 = self.emb_1st(pos).unsqueeze(-1)
        embpos2 = self.emb_context(pos).unsqueeze(-1)
        embneg1 = self.emb_1st(neg)
        embneg2 = self.emb_context(neg)
        pos1 = emb1 @ embpos1
        pos2 = emb2 @ embpos2
        neg1 = emb1 @ embneg1.mT
        neg2 = emb2 @ embneg2.mT
        return pos1.squeeze(), pos2.squeeze(), neg1.squeeze(-2), neg2.squeeze(-2)

    @staticmethod
    def triplet_loss(pos1, pos2, neg1, neg2):
        """
        三元组损失函数，输入的4个值分别是：
        dot(center, pos) 1阶相似性、2阶相似性、dot(center, neg) 1阶相似性、2阶相似性
        """
        logsigm = nn.LogSigmoid()
        neg_loss = -logsigm(-neg1) - logsigm(-neg2)
        _loss = -logsigm(pos1) - logsigm(pos2) + torch.sum(neg_loss, dim=-1)
        return torch.mean(_loss)

    @torch.inference_mode()
    def embedding(self, x=None):
        flag = self.training
        self.eval()
        if x is None:
            x = torch.arange(0, self.num_road, dtype=torch.long, device=DEVICE)
        else:
            x = torch.tensor(x, dtype=torch.long, device=DEVICE)
        emb = torch.cat([self.emb_1st(x), self.emb_2nd(x)], dim=-1)
        if flag:
            self.train()
        return emb


class GATLayer(nn.Module):
    """简单的GAT层+多头注意力融合"""
    def __init__(self, input_dim, emb_dim=32, gconv=None, num_heads=4, **gconv_kwargs):
        super(GATLayer, self).__init__()
        conv = {'gat': dglnn.GATConv, 'dotgat': dglnn.DotGatConv}[gconv]
        self.conv = conv(input_dim, emb_dim, num_heads, **gconv_kwargs)
        self.linear = nn.Linear(emb_dim * num_heads, emb_dim)

    def forward(self, graph, feat):
        feat = self.conv(graph, feat)
        feat = feat.flatten(start_dim=1)
        return self.linear(feat)


class VGAE(nn.Module):
    """
    图变分自编码器，用于训练结点（研究中等价于路段）的表示向量。
    模型的优化目标有2个：1.表示向量的内积能够反映连边预测特征；2.表示向量分布为多维正态分布。
    模型结构：encoder是多层的图卷积，decoder就是简单的内积。

    :param input_dim: 模型输入的特征数（结点初始特征维度）
    :param emb_dim: 中间的嵌入向量维度。encoder输出时，mean和var都是这个维度
    :param layer_num: encoder中图卷积的层数
    :param gconv: encoder中图卷积的实现方式，支持'gc'(GraphConv), 'sage'(SAGEConv)
    :param gconv_kwargs: 初始化图卷积层的其他参数，`dict`对象
    :param activation: encoder中卷积层之间的激活函数，可以是str, nn.Module, function
    """
    def __init__(self, input_dim, emb_dim=32, layer_num=2, gconv='gc',
                 gconv_kwargs=None, activation='relu', dropout=0.2):
        super(VGAE, self).__init__()
        conv = {'gc': dglnn.GraphConv, 'sage': dglnn.SAGEConv}[gconv]
        gconv_kwargs = {} if gconv_kwargs is None else gconv_kwargs
        encoder = [conv(input_dim, emb_dim, **gconv_kwargs)]
        for i in range(layer_num - 2):
            encoder.append(conv(emb_dim, emb_dim, **gconv_kwargs))
        encoder.append(conv(emb_dim, 2 * emb_dim, **gconv_kwargs))

        self.emb_dim = emb_dim
        self.encoder = nn.ModuleList(encoder)
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mfgs, x, positive_graph, negative_graph):
        """
        包含生成结点表示向量的mu、重采样、连边预测流程

        :param mfgs: message flow graphs
        :param x: 输入的初始特征
        :param positive_graph: 正样本子图
        :param negative_graph: 负样本子图
        :return (pos, neg, mu, sigma) = (正样本预测logit, 负样本预测logit, 嵌入向量的mu, 嵌入向量的sigma)
        """
        # 生成结点的表示向量
        mu, sigma = self.encode(mfgs, x)
        z = mu + sigma * torch.normal(0.0, 1.0, size=sigma.size(), device=DEVICE)
        # 连边预测，使用dot product
        pos = self.dot_predict(positive_graph, z)
        neg = self.dot_predict(negative_graph, z)
        return pos, neg, mu, sigma

    def encode(self, mfgs, x):
        h = x
        for i in range(len(self.encoder)):
            h_dst = h[: mfgs[i].num_dst_nodes()]
            h = self.encoder[i](mfgs[i], (h, h_dst))
            if i != len(self.encoder) - 1:
                h = self.dropout(h)
                h = self.activation(h)
        mu, sigma = torch.split(h, self.emb_dim, dim=-1)
        sigma = torch.exp(sigma / 2)
        return mu, sigma

    @staticmethod
    def dot_predict(sub_graph: dgl.DGLGraph, emb):
        """连边预测，使用dot product"""
        with sub_graph.local_scope():
            sub_graph.ndata['z'] = emb
            sub_graph.apply_edges(dgl.function.u_dot_v('z', 'z', 'score'))
            return torch.squeeze(sub_graph.edata['score'], dim=-1)

    @staticmethod
    def reconstruct_loss(pos, neg):
        """重建损失：正确预测连边"""
        logsigm = nn.LogSigmoid()
        return -logsigm(pos).mean() - logsigm(-neg).sum() / pos.size(0)

    @staticmethod
    def kldiv_loss(mu, sigma):
        """分布差异损失：正态分布假设下的KL散度"""
        loss = torch.square(mu) + torch.square(sigma) - \
               torch.log(1e-8 + torch.square(sigma)) - 1
        return 0.5 * torch.mean(loss.sum(dim=-1))

    @torch.inference_mode()
    def embedding(self, graph, x):
        flag = self.training
        self.eval()
        mu, sigma = self.encode([graph] * len(self.encoder), x)
        if flag:
            self.train()
        return mu


class VGAE_NoBatch(VGAE):
    """
    图变分自编码器，用于训练结点（研究中等价于路段）的表示向量。
    该模型以谱卷积SGConv为卷积层，不支持batch操作

    :param input_dim: 模型输入的特征数（结点初始特征维度）
    :param emb_dim: 中间的嵌入向量维度。encoder输出时，mean和var都是这个维度
    :param layer_num: encoder中图卷积的层数
    :param gconv: encoder中图卷积的实现方式，支持'gc', 'sage', 'sgc', 'gat'
    :param gconv_kwargs: 初始化图卷积层的其他参数，`dict`对象
    :param activation: encoder中卷积层之间的激活函数，可以是str, nn.Module, function
    """
    def __init__(self, input_dim, emb_dim=32, layer_num=2, gconv='sgc',
                 gconv_kwargs=None, activation='relu', dropout=0.2):
        super(VGAE, self).__init__()
        conv = {'gc': dglnn.GraphConv, 'sage': dglnn.SAGEConv, 'sgc': dglnn.SGConv,
                'gat': GATLayer, 'dotgat': GATLayer}[gconv]
        gconv_kwargs = {} if gconv_kwargs is None else gconv_kwargs
        if gconv in ('gat', 'dotgat'):
            gconv_kwargs.update(gconv=gconv)
        encoder = [conv(input_dim, emb_dim, **gconv_kwargs)]
        for i in range(layer_num - 2):
            encoder.append(conv(emb_dim, emb_dim, **gconv_kwargs))
        encoder.append(conv(emb_dim, 2 * emb_dim, **gconv_kwargs))

        self.emb_dim = emb_dim
        self.encoder = nn.ModuleList(encoder)
        self.activation = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def encode(self, mfgs, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](mfgs, x)
            if i != len(self.encoder) - 1:
                x = self.dropout(x)
                x = self.activation(x)
        mu, sigma = torch.split(x, self.emb_dim, dim=-1)
        sigma = torch.exp(sigma / 2)
        return mu, sigma

    @torch.inference_mode()
    def embedding(self, graph, x):
        flag = self.training
        self.eval()
        mu, sigma = self.encode(graph, x)
        if flag:
            self.train()
        return mu


def train_LINE(input_file, output_path, emb_dim=32,
               neg_sample=10, batch_size=1024, epoch=200):
    """
    从路网拓扑结构中，使用LINE模型训练路段的表示向量

    :param emb_dim: 嵌入向量的维度。是<1st_emb, 2nd_emb>拼接后的总长度
    :param neg_sample: 负样本采样数量，对每个正样本随机取k个负样本
    """
    # 准备数据、模型、损失函数、优化器
    data = RoadNetworkDataLINE(input_file)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    model = LINE(data.num_roads, emb_dim).to(DEVICE)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for i in range(epoch):
        ep_loss = []
        pbar = tqdm(dataloader)
        for batch in pbar:
            neg = data.batch_negative_sample(batch[0].shape[0], neg_sample)
            pred = model(*batch, neg)
            loss = model.triplet_loss(*pred)
            pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}')
            ep_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.close()
        ep_loss = torch.mean(torch.stack(ep_loss))
        print(f'epoch {i:>3d}: loss = {ep_loss.item():>7f}', flush=True)
        history.append({'epoch': i, 'loss': ep_loss.item()})

    # 保存训练结果
    history = pd.DataFrame(history)
    history.to_csv(Path(output_path).with_suffix('.csv'), index=False)
    np.save(Path(output_path).with_suffix('.npy'),
            model.embedding().cpu().numpy())
    return model, history


def train_VGAE_batch(input_file, output_path, num_layer=2, emb_dim=32,
                     neg_sample=10, batch_size=1024, epoch=500, kldiv_ratio=1,
                     gconv='gc', gconv_args=None, activation='relu'):
    """
    从路网拓扑结构中，使用Graph VAE模型训练路段的表示向量

    :param num_layer: encoder中GCN卷积层的数量
    :param emb_dim: 嵌入向量的维度
    :param neg_sample: 负样本采样数量，对每个正样本随机取k个负样本
    :param kldiv_ratio: KL散度损失函数的权重。总损失 = reconst + `ratio` * kldiv
    :param gconv: encoder中图卷积的实现方式，支持'gc'(GraphConv), 'sage'(SAGEConv)
    :param gconv_args: 初始化图卷积层的其他参数，`dict`对象
    :param activation: encoder中卷积层之间的激活函数，可以是str, nn.Module, function
    """
    graph, dataloader = create_graph_data_GAE(input_file, num_layer, neg_sample, batch_size)
    model = VGAE(graph.ndata['feat'].size(1), emb_dim, num_layer,
                 gconv, gconv_args, activation).to(DEVICE)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for i in range(epoch):
        ep_loss = []
        pbar = tqdm(dataloader)
        for input_nodes, positive_graph, negative_graph, blocks in pbar:
            pos, neg, mu, sigma = model(blocks, blocks[0].srcdata['feat'],
                                        positive_graph, negative_graph)
            reconst_loss = model.reconstruct_loss(pos, neg)
            kldiv = kldiv_ratio * model.kldiv_loss(mu, sigma)
            loss = reconst_loss + kldiv
            pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}, '
                                 f'reconstruction: {reconst_loss.item():>7f}, '
                                 f'KL divergence: {kldiv.item():>7f}')
            ep_loss.append((loss.item(), reconst_loss.item(), kldiv.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.close()
        ep_loss = torch.mean(torch.tensor(ep_loss), dim=0)
        print(f'epoch {i:>3d}: loss = {ep_loss[0].item():>7f},',
              f'reconstruction = {ep_loss[1].item():>7f},',
              f'KL divergence = {ep_loss[2].item():>7f}', flush=True)
        history.append({'epoch': i, 'loss': ep_loss[0].item(),
                        'reconstruction': ep_loss[1].item(),
                        'kl divergence': ep_loss[2].item()})

    # 保存训练结果
    history = pd.DataFrame(history)
    history.to_csv(Path(output_path).with_suffix('.csv'), index=False)
    np.save(Path(output_path).with_suffix('.npy'),
            model.embedding(graph, graph.ndata['feat']).cpu().numpy())
    return model, history


def train_VGAE_nobatch(input_file, output_path, num_layer=2, emb_dim=32, neg_sample=10,
                       epoch=500, kldiv_ratio=1, gconv='sgc', gconv_args=None, activation='relu'):
    """
    从路网拓扑结构中，使用Graph VAE模型训练路段的表示向量。
    这个函数使用SGConv谱卷积，没有batch的直推式训练

    :param num_layer: encoder中GCN卷积层的数量
    :param emb_dim: 嵌入向量的维度
    :param neg_sample: 负样本采样数量，对每个正样本随机取k个负样本
    :param kldiv_ratio: KL散度损失函数的权重。总损失 = reconst + `ratio` * kldiv
    :param gconv: encoder中图卷积的实现方式。详见`VGAE_NoBatch`的`gconv`参数
    :param gconv_args: 初始化图卷积层的其他参数，`dict`对象
    :param activation: encoder中卷积层之间的激活函数，可以是str, nn.Module, function
    """
    graph, dataloader = create_graph_data_GAE(input_file, num_layer, neg_sample, 100000)
    model = VGAE_NoBatch(graph.ndata['feat'].size(1), emb_dim, num_layer,
                         gconv, gconv_args, activation).to(DEVICE)
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for i in range(epoch):
        neg_graph = dgl.rand_graph(graph.num_nodes(), graph.num_edges() * neg_sample)
        neg_graph = dgl.to_bidirected(neg_graph).to(DEVICE)
        pos, neg, mu, sigma = model(graph, graph.ndata['feat'], graph, neg_graph)

        reconstr = model.reconstruct_loss(pos, neg)
        kldiv = kldiv_ratio * model.kldiv_loss(mu, sigma)
        loss = reconstr + kldiv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {i:>3d}: loss = {loss.item():>7f},',
              f'reconstruction = {reconstr.item():>7f},',
              f'KL divergence = {kldiv.item():>7f}', flush=True)
        history.append({'epoch': i, 'loss': loss.item(),
                        'reconstruction': reconstr.item(),
                        'kl divergence': kldiv.item()})

    # 保存训练结果
    history = pd.DataFrame(history)
    history.to_csv(Path(output_path).with_suffix('.csv'), index=False)
    np.save(Path(output_path).with_suffix('.npy'),
            model.embedding(graph, graph.ndata['feat']).cpu().numpy())
    return model, history


if __name__ == '__main__':
    train_LINE(TRAJ_ROOT / 'roads/road_1_roadlink.csv',
               TRAJ_ROOT / 'embedding/roademb_line_d32_neg5_ep500',
               emb_dim=32, neg_sample=5, epoch=500)
    # train_VGAE_batch(TRAJ_ROOT / 'roads/road_1_roadlink.csv',
    #                  TRAJ_ROOT / 'embedding/roademb_vgae_d32_neg10_ep200',
    #                  emb_dim=32, neg_sample=10, epoch=500)
    # train_VGAE_nobatch(TRAJ_ROOT / 'roads/road_1_roadlink.csv',
    #                    TRAJ_ROOT / 'embedding/roademb_vgae_gat_d32_neg10_ep500',
    #                    emb_dim=32, neg_sample=10, epoch=500, kldiv_ratio=1,
    #                    gconv='gat', gconv_args={'num_heads': 4})
