# 轨迹目的地预测模型：模型训练、精度验证流程和方法


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from functools import partial
from globalval import *
from predict_dataset import *
from predict_nn import *


assert torch.cuda.is_available()
DEVICE = torch.device("cuda:0")
TQDM_UPDATE_TIME = 0.1


def set_module_param(_DEVICE=None, _TQDM_UPDATE_TIME=None):
    """修改本模块的全局变量值"""
    global DEVICE, TQDM_UPDATE_TIME
    if _DEVICE is not None:
        DEVICE = _DEVICE
    if _TQDM_UPDATE_TIME is not None:
        TQDM_UPDATE_TIME = _TQDM_UPDATE_TIME


class MaskedCrossEntr:
    """
    适用于mask的交叉熵，True值表示pad

    :param clip_start: 开头一段长度的轨迹不参与预测
    """
    def __init__(self, class_weight=None, clip_start=10):
        self.fn = nn.CrossEntropyLoss(class_weight, reduction='none')
        self.clip_start = clip_start

    def __call__(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        :param input: size = (bsize, Len, C)
        :param target: size = (bsize,)
        :param mask: size = (bsize, Len)
        """
        if self.clip_start:
            _len = input.size(1)
            _, input = input.split([self.clip_start, _len - self.clip_start], dim=1)
            _, mask = mask.split([self.clip_start, _len - self.clip_start], dim=1)
        input = input.transpose(1, 2)
        target = target.unsqueeze(-1).expand(mask.size())
        target = torch.where(mask, self.fn.ignore_index, target)
        weight = torch.count_nonzero(~mask, dim=1).unsqueeze(1) / mask.size(1)
        return torch.nanmean(self.fn(input, target) / weight)


class MaskedMSE:
    """
    适用于mask的L2损失，True值表示pad

    :param clip_start: 开头一段长度的轨迹不参与预测
    :param static_target: True表示target是静态的（不随时间维改变）
    """
    def __init__(self, clip_start=10, static_target=True):
        self.fn = nn.MSELoss(reduction='none')
        self.clip_start = clip_start
        self.static_target = static_target

    def __call__(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        :param input: size = (bsize, Len, d)
        :param target: size = (bsize, d) if static_target else (bsize, Len, d)
        :param mask: size = (bsize, Len)
        """
        if self.clip_start:
            _len = input.size(1)
            _, input = input.split([self.clip_start, _len - self.clip_start], dim=1)
            _, mask = mask.split([self.clip_start, _len - self.clip_start], dim=1)
            if not self.static_target:
                _, target = target.split([self.clip_start, _len - self.clip_start], dim=1)
        if self.static_target:
            target = target.unsqueeze(1).expand(-1, input.size(1), -1)
        mask = mask.unsqueeze(-1).expand(-1, -1, input.size(-1))
        loss = self.fn(input, target)
        weight = torch.count_nonzero(~mask, dim=1).unsqueeze(1) / mask.size(1)
        return torch.nanmean(torch.where(mask, 0, loss) / weight)


class MaskedQuantileLoss:
    """分位数损失函数"""
    def __init__(self, quantile, clip_start=10):
        self.q = quantile
        self.clip_start = clip_start
        self.max0 = nn.ReLU()

    def __call__(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        :param input: size = (bsize, Len, d)
        :param target: size = (bsize, d)
        :param mask: size = (bsize, Len)
        """
        if self.clip_start:
            _len = input.size(1)
            _, input = input.split([self.clip_start, _len - self.clip_start], dim=1)
            _, mask = mask.split([self.clip_start, _len - self.clip_start], dim=1)
        target = target.unsqueeze(1).expand(-1, input.size(1), -1)
        mask = mask.unsqueeze(-1).expand(-1, -1, input.size(-1))
        weight = torch.count_nonzero(~mask, dim=1).unsqueeze(1) / mask.size(1)
        loss = self.q * self.max0(target - input) + \
               (1 - self.q) * self.max0(input - target)
        return torch.nanmean(torch.where(mask, 0, loss) / weight)


def empty_loss(*x):
    return torch.zeros((), device=DEVICE)


def accuracy_(pred: torch.Tensor, y: torch.Tensor):
    """模型预测精度，相当于acc@1，预测概率最高的路段是否正确"""
    acc = torch.count_nonzero(pred.argmax(-1) == y) / y.numel()
    return acc.item()


def accuracyK_(pred: torch.Tensor, y: torch.Tensor, k=5):
    """模型预测精度，acc@k，预测概率前K高的路段中，是否有真实标签"""
    topk = pred.topk(k, dim=-1).indices
    topk = torch.any(topk == y.unsqueeze(-1), dim=-1)
    return (topk.count_nonzero() / topk.numel()).item()


def f1score_(pred: torch.Tensor, y: torch.Tensor):
    """
    模型预测weighted macro-F1得分：
    F1 = 2 * precision * recall / (precision + recall)
    """
    pred_np = pred.argmax(-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return f1_score(y_np, pred_np, average='weighted').item()


def mrr_(pred: torch.Tensor, y: torch.Tensor):
    """
    模型的MRR(mean reciprocal rank)指标，
    MRR = mean(1 / argsort(pred)[y])
    """
    rank = pred.argsort(-1, descending=True) == y.unsqueeze(-1)
    weight = 1 / torch.arange(1, rank.size(-1) + 1, dtype=torch.float,
                              device=rank.device)
    return torch.masked_select(weight.expand(1, 1, -1), rank).mean().item()


_resolution = 2 / max(RESEARCH_BOUND[2] - RESEARCH_BOUND[0],
                      RESEARCH_BOUND[3] - RESEARCH_BOUND[1])


def _xy_distance(pred: torch.Tensor, y: torch.Tensor):
    return torch.square(pred - y).sum(dim=-1).sqrt()


def mean_dist_(pred: torch.Tensor, y: torch.Tensor):
    """预测目的地xy坐标与真实值的距离，取平均值，转换成单位meter"""
    xydist = _xy_distance(pred, y).mean().item()
    return xydist / _resolution


class BaseMedianFn:
    def __init__(self):
        self.data = []

    def median(self):
        data = torch.cat(self.data, dim=0).median()
        self.data = []
        return data.item()


class MedianDist(BaseMedianFn):
    """预测目的地xy坐标与真实值的距离，取中位数，转换成单位meter"""
    def __call__(self, pred: torch.Tensor, y: torch.Tensor):
        geodist = _xy_distance(pred, y) / _resolution
        self.data.append(geodist.detach().cpu())
        return geodist.median().item()


def mre_(pred: torch.Tensor, y: torch.Tensor, length: torch.Tensor):
    """mean relative error = mean(||pred - y|| / length)"""
    relative_err = _xy_distance(pred, y) / length.squeeze(-1)
    return relative_err.mean().item()


class MedianRelErr(BaseMedianFn):
    """median relative error = median(||pred - y|| / length)"""
    def __call__(self, pred: torch.Tensor, y: torch.Tensor, length: torch.Tensor):
        relative_err = _xy_distance(pred, y) / length.squeeze(-1)
        self.data.append(relative_err.detach().cpu())
        return relative_err.median().item()


def accuracy_range_(pred: torch.Tensor, y: torch.Tensor, distance=1000.0):
    """准确率区间，计算batch中预测误差小于distance (meter)的样本比例"""
    geodist = _xy_distance(pred, y) / _resolution
    return (torch.count_nonzero(geodist <= distance) / geodist.numel()).item()


def train_model(train_data: TrajectoryData, model: nn.Module, batch_size=128,
                epoch=100, lr=1e-3, val_data: TrajectoryData = None, val_freq=5,
                train_mode='colearn', loss2_gamma=1.0, output_path=None):
    """
    训练轨迹目的地预测模型，并验证精度。
    预测目标：可能1：仅有目的地所在路段，没有多任务联合学习。
             可能2：多任务联合学习，预测目标包括目的地所在路段、轨迹出行时间。

    :param train_data: 使用的训练集
    :param model: 使用的预测模型
    :param val_data: 验证集，用于每`val_freq`个epoch之后验证精度指标
    :param val_freq: 使用验证集确认精度的频率，即间隔`val_freq`个epoch后检查一次
    :param train_mode: 调用的训练函数：
                       'colearn': 使用多任务联合学习，预测目的地的同时预测出行用时
                       'no_colearn': 仅预测目的地所在路段
                       'diffusion': 扩散模型训练
                       'autoreg': 以自回归的方式训练，每个定位点都输出一个预测值
                       带有'_xy'后缀: 输出目的地是回归xy值，不是分类的路段概率
                       'autoreg3_xy': 每组因变量输出3个值：置信区间min, 期望值, 置信区间max
    """
    assert train_mode in ['colearn', 'no_colearn', 'diffusion', 'autoreg', 'colearn_xy', 'autoreg_xy',
                          'autoreg3_xy', 'diffusion_xy', 'diffusion_autoreg_xy',
                          'no_colearn_xy', 'no_colearn_autoreg_xy']

    _num_workers = 0
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate,
                              num_workers=_num_workers, pin_memory=True,
                              persistent_workers=_num_workers > 0)
    val_fn, val_fn2 = {}, {}
    if val_data is not None:
        _num_workers = 0
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=val_data.collate,
                                num_workers=_num_workers, pin_memory=True,
                                persistent_workers=_num_workers > 0)
        if train_mode.endswith('_xy'):
            val_fn.update(mean_dist=mean_dist_, acc_500m=partial(accuracy_range_, distance=500),
                          acc_1km=partial(accuracy_range_, distance=1000))
        else:
            val_fn.update(acc1=accuracy_, acc5=partial(accuracyK_, k=5),
                          acc10=partial(accuracyK_, k=10), f1score=f1score_, mrr=mrr_)
    else:
        val_loader = None
    history = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if train_mode == 'colearn':
        loss_fn1 = nn.CrossEntropyLoss(_to_device(train_data.class_weight, DEVICE))
        val_fn2.update(val_time_loss=nn.MSELoss())
        loss_fn2 = nn.MSELoss()
    elif train_mode == 'no_colearn':
        loss_fn1 = nn.CrossEntropyLoss(_to_device(train_data.class_weight, DEVICE))
        loss_fn2 = None
    elif train_mode == 'diffusion':
        loss_fn1 = nn.MSELoss()
        loss_fn2 = nn.CrossEntropyLoss(_to_device(train_data.class_weight, DEVICE))
    elif train_mode == 'autoreg':
        loss_fn1 = MaskedCrossEntr(_to_device(train_data.class_weight, DEVICE))
        loss_fn2 = MaskedMSE()
    elif train_mode == 'colearn_xy':
        loss_fn1 = nn.MSELoss()
        val_fn2.update(val_time_loss=nn.MSELoss())
        loss_fn2 = nn.MSELoss()
    elif train_mode == 'no_colearn_xy':
        # TODO: LSI_LSTM训练用的是MAE loss
        loss_fn1 = nn.L1Loss()
        loss_fn2 = None
    elif train_mode == 'autoreg_xy':
        loss_fn1 = MaskedMSE()
        loss_fn2 = MaskedMSE()
    elif train_mode == 'no_colearn_autoreg_xy':
        # TODO: 仅限DND2019去除重复路段后，只截取5个之后的
        loss_fn1 = MaskedMSE(clip_start=10)
        loss_fn2 = empty_loss
    elif train_mode == 'autoreg3_xy':
        loss_fn1 = [MaskedQuantileLoss(0.1), MaskedQuantileLoss(0.5), MaskedQuantileLoss(0.9), MaskedMSE()]
        loss_fn2 = [MaskedMSE() for _ in range(4)]
        assert getattr(model, 'y_num', None) == len(loss_fn1)
    elif train_mode == 'diffusion_xy':
        loss_fn1 = nn.MSELoss()
        loss_fn2 = None
    elif train_mode == 'diffusion_autoreg_xy':
        loss_fn1 = MaskedMSE(static_target=False)
        loss_fn2 = None

    try:
        if train_mode == 'colearn' or train_mode == 'colearn_xy':
            model, history = _train_model_colearn(
                train_loader, model, epoch, history,
                loss_fn1, loss_fn2, loss2_gamma, optimizer,
                val_loader, val_freq, val_fn, val_fn2)
        elif train_mode == 'no_colearn' or train_mode == 'no_colearn_xy':
            model, history = _train_model_no_colearn(
                train_loader, model, epoch, history,
                loss_fn1, optimizer, val_loader, val_freq, val_fn)
        elif train_mode == 'diffusion' or train_mode == 'diffusion_xy':
            model, history = _train_model_diffusion(
                train_loader, model, epoch, history,
                loss_fn1, loss_fn2, loss2_gamma, optimizer,
                val_loader, val_freq, val_fn)
        elif train_mode in ['autoreg', 'autoreg_xy', 'autoreg3_xy', 'no_colearn_autoreg_xy']:
            model, history = _train_model_autoreg(
                train_loader, model, epoch, history,
                loss_fn1, loss_fn2, loss2_gamma, optimizer,
                val_loader, val_freq, val_fn, val_fn2)
        elif train_mode == 'diffusion_autoreg_xy':
            model, history = _train_model_diffusion_autoreg(
                train_loader, model, epoch, history,
                loss_fn1, loss_fn2, loss2_gamma, optimizer,
                val_loader, val_freq, val_fn)
    except KeyboardInterrupt:
        pass
    finally:
        # 保存训练结果
        if output_path is not None:
            output_path = Path(output_path)
            history = pd.DataFrame(history)
            history.to_csv(output_path.with_suffix('.csv'), index=False)
            torch.save(model, output_path.with_suffix('.pth'))
    return model, history


def _to_device(obj, device):
    """用于obj递归迁移数据到某个设备上"""
    if obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    _func = partial(_to_device, device=device)
    if isinstance(obj, list):
        return list(map(_func, obj))
    return tuple(map(_func, obj))


def _train_model_no_colearn(train_loader: DataLoader, model: nn.Module, epoch: int,
                            history, loss_fn1, optimizer: torch.optim.Optimizer,
                            val_loader: DataLoader, val_freq: int, val_fn):
    """用于单任务训练：仅预测轨迹目的地"""
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                pred = model(*x)
                loss = loss_fn1(pred, y[0])
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}',
                                     refresh=False)
                ep_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss.item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss.item()}

        # 验证集验证精度
        if val_loader is not None and (i + 1) % val_freq == 0:
            ep_acc = []
            model.eval()
            with tqdm(val_loader, desc='validate', mininterval=TQDM_UPDATE_TIME) as pbar, \
                    torch.inference_mode():
                for batch in pbar:
                    x, y = _to_device(batch, DEVICE)
                    pred = model(*x)
                    acc = tuple(fn(pred, y[0]) for _, fn in val_fn.items())
                    ep_acc.append(acc)
                ep_acc = torch.tensor(ep_acc).mean(dim=0)
                ep_str.extend(f'{name} = {ep_acc[i].item()}' for i, name in enumerate(val_fn))
                ep_dict.update({name: ep_acc[i].item() for i, name in enumerate(val_fn)})
            model.train()

        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    return model, history


def _train_model_colearn(train_loader: DataLoader, model: nn.Module, epoch: int,
                         history, loss_fn1, loss_fn2, loss2_gamma, optimizer: torch.optim.Optimizer,
                         val_loader: DataLoader, val_freq: int, val_fn1, val_fn2):
    """用于多任务训练：轨迹目的地+出行用时"""
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                pred = model(*x)
                loss_entr = loss_fn1(pred[0], y[0])
                loss_time = loss2_gamma * loss_fn2(pred[1], y[1])
                loss = loss_entr + loss_time
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}, '
                                     f'cross_entropy: {loss_entr.item():>7f}, '
                                     f'time_loss: {loss_time.item():>7f}', refresh=False)
                ep_loss.append((loss.item(), loss_entr.item(), loss_time.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss[0].item():>7f}', f'loss_entr = {ep_loss[1].item():>7f}',
                  f'time_loss = {ep_loss[2].item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss[0].item(), 'loss_entr': ep_loss[1].item(),
                   'time_loss': ep_loss[2].item()}

        # 验证集验证精度
        if val_loader is not None and (i + 1) % val_freq == 0:
            ep_acc1, ep_acc2 = [], []
            model.eval()
            with tqdm(val_loader, desc='validate', mininterval=TQDM_UPDATE_TIME) as pbar, \
                    torch.inference_mode():
                for batch in pbar:
                    x, y = _to_device(batch, DEVICE)
                    pred = model(*x)
                    ep_acc1.append(tuple(fn(pred[0], y[0]) for _, fn in val_fn1.items()))
                    ep_acc2.append(tuple(fn(pred[1], y[1]) for _, fn in val_fn2.items()))
                ep_acc1 = torch.tensor(ep_acc1).mean(dim=0)
                ep_acc2 = torch.tensor(ep_acc2).mean(dim=0)
                ep_str.extend(f'{name} = {ep_acc1[i].item():>7f}' for i, name in enumerate(val_fn1))
                ep_str.extend(f'{name} = {ep_acc2[i].item():>7f}' for i, name in enumerate(val_fn2))
                ep_dict.update({name: ep_acc1[i].item() for i, name in enumerate(val_fn1)})
                ep_dict.update({name: ep_acc2[i].item() for i, name in enumerate(val_fn2)})
            model.train()

        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    return model, history


def _train_model_autoreg(train_loader: DataLoader, model: nn.Module, epoch: int,
                         history, loss_fn1, loss_fn2, loss2_gamma, optimizer: torch.optim.Optimizer,
                         val_loader: DataLoader, val_freq: int, val_fn1, val_fn2):
    """用自回归的方式训练模型，同样是多任务的方式（和_colearn相似）"""
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                target_road, trip_time, _, _ = model(*x, output_seq=True)
                if isinstance(loss_fn1, (list, tuple)):
                    target_roads = target_road.chunk(len(loss_fn1), dim=-1)
                    trip_times = trip_time.chunk(len(loss_fn2), dim=-1)
                    loss_entr = torch.stack([_fn1(_road, y[0], x[4])
                                             for _fn1, _road in zip(loss_fn1, target_roads)]).mean()
                    loss_time = loss2_gamma * torch.stack([_fn2(_time, y[1], x[4])
                                                           for _fn2, _time in zip(loss_fn2, trip_times)]).mean()
                else:
                    loss_entr = loss_fn1(target_road, y[0], x[4])
                    loss_time = loss2_gamma * loss_fn2(trip_time, y[1], x[4])
                loss = loss_entr + loss_time
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}, '
                                     f'cross_entropy: {loss_entr.item():>7f}, '
                                     f'time_loss: {loss_time.item():>7f}', refresh=False)
                ep_loss.append((loss.item(), loss_entr.item(), loss_time.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss[0].item():>7f}', f'loss_entr = {ep_loss[1].item():>7f}',
                  f'time_loss = {ep_loss[2].item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss[0].item(), 'loss_entr': ep_loss[1].item(),
                   'time_loss': ep_loss[2].item()}

        # 验证集验证精度
        if val_loader is not None and (i + 1) % val_freq == 0:
            ep_acc1, ep_acc2 = [], []
            model.eval()
            with tqdm(val_loader, desc='validate', mininterval=TQDM_UPDATE_TIME) as pbar, \
                    torch.inference_mode():
                for batch in pbar:
                    x, y = _to_device(batch, DEVICE)
                    pred = model(*x, predict_final=True, pred_y_index=1)
                    seq_len = target_road.size(1)
                    _, seq_mask = seq_mask.split([seq_mask.size(1) - seq_len, seq_len], dim=1)
                    ep_acc1.append(tuple(fn(pred[0], y[0]) for _, fn in val_fn1.items()))
                    ep_acc2.append(tuple(fn(pred[1], y[1]) for _, fn in val_fn2.items()))
                ep_acc1 = torch.tensor(ep_acc1).mean(dim=0)
                ep_acc2 = torch.tensor(ep_acc2).mean(dim=0)
                ep_str.extend(f'{name} = {ep_acc1[i].item():>7f}' for i, name in enumerate(val_fn1))
                ep_str.extend(f'{name} = {ep_acc2[i].item():>7f}' for i, name in enumerate(val_fn2))
                ep_dict.update({name: ep_acc1[i].item() for i, name in enumerate(val_fn1)})
                ep_dict.update({name: ep_acc2[i].item() for i, name in enumerate(val_fn2)})
            model.train()

        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    return model, history


def _train_model_diffusion(train_loader: DataLoader, model: DiffusionPredictor, epoch: int,
                           history, loss_fn1, loss_fn2, loss2_gamma, optimizer: torch.optim.Optimizer,
                           val_loader: DataLoader, val_freq: int, val_fn):
    """用于扩散模型的训练：预测噪声"""
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                t = torch.randint(low=1, high=model.diff_steps + 1,
                                  size=(y[0].size(0),), device=DEVICE)
                with torch.no_grad():
                    cond_pred = model.encoder(*x, output_seq=True, predict_final=True, pred_y_index=1)
                e = torch.randn_like(y[0], device=DEVICE)
                y_t = model.q_sample(y[0], cond_pred[0], t, e)
                noise_pred = model(cond_pred[2], cond_pred[3], x[4], cond_pred[0], y_t, t)
                loss_noise = loss_fn1(noise_pred, e)
                # y0_prob = model.reparameter_y0_hat(y_t, cond_pred[0], noise_pred, t)
                # y0_prob = - ((y0_prob - 1) ** 2)
                # loss_ce = loss2_gamma * loss_fn2(y0_prob, y[0])
                loss = loss_noise   # + loss_ce
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}, ', refresh=False)
                                     # f'noise_mse: {loss_noise.item():>7f}, '
                                     # f'cross_entropy: {loss_ce.item():>7f}', refresh=False)
                ep_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss.item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss.item()}

        # 验证集验证精度
        if val_loader is not None and (i + 1) % val_freq == 0:
            ep_acc = []
            model.eval()
            with tqdm(val_loader, desc='validate', mininterval=TQDM_UPDATE_TIME) as pbar, \
                    torch.inference_mode():
                for batch in pbar:
                    x, y = _to_device(batch, DEVICE)
                    pred = model.p_sample_loop_repeat(*x)
                    acc = tuple(fn(pred, y[0]) for _, fn in val_fn.items())
                    ep_acc.append(acc)
                ep_acc = torch.tensor(ep_acc).mean(dim=0)
                ep_str.extend(f'{name} = {ep_acc[i].item()}' for i, name in enumerate(val_fn))
                ep_dict.update({name: ep_acc[i].item() for i, name in enumerate(val_fn)})
            model.train()
            model.encoder.eval()

        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    return model, history


def _train_model_diffusion_autoreg(
        train_loader: DataLoader, model: DiffusionPredictor, epoch: int,
        history, loss_fn1, loss_fn2, loss2_gamma, optimizer: torch.optim.Optimizer,
        val_loader: DataLoader, val_freq: int, val_fn):
    """用于扩散模型的训练 + 自回归方式训练序列"""
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                t = torch.randint(low=1, high=model.diff_steps + 1,
                                  size=(y[0].size(0),), device=DEVICE)
                with torch.no_grad():
                    cond_pred = model.encoder(*x, output_seq=True, pred_y_index=1)
                    cond_pred = (cond_pred[0].transpose(0, 1), *cond_pred[1:])
                y_true = y[0].unsqueeze(0).expand(cond_pred[0].size(0), -1, -1)
                e = torch.randn_like(y_true, device=DEVICE)
                y_t = model.q_sample(y_true, cond_pred[0], t, e)
                noise_pred = model(cond_pred[2], cond_pred[3], x[4], cond_pred[0], y_t, t)
                loss = loss_fn1(noise_pred.transpose(0, 1), e.transpose(0, 1), x[4])
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}, ', refresh=False)
                ep_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss.item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss.item()}

        # 验证集验证精度
        if val_loader is not None and (i + 1) % val_freq == 0:
            ep_acc = []
            model.eval()
            with tqdm(val_loader, desc='validate', mininterval=TQDM_UPDATE_TIME) as pbar, \
                    torch.inference_mode():
                for batch in pbar:
                    x, y = _to_device(batch, DEVICE)
                    pred = model.p_sample_loop_repeat(*x)
                    acc = tuple(fn(pred, y[0]) for _, fn in val_fn.items())
                    ep_acc.append(acc)
                ep_acc = torch.tensor(ep_acc).mean(dim=0)
                ep_str.extend(f'{name} = {ep_acc[i].item()}' for i, name in enumerate(val_fn))
                ep_dict.update({name: ep_acc[i].item() for i, name in enumerate(val_fn)})
            model.train()
            model.encoder.eval()

        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    return model, history


class TestRecorder:
    """
    用于记录模型测试时的输出，以便后续分析精度指标
    """
    def __init__(self):
        self.y_pred = []
        self.y_true = []
        self.others = []

    def add_batch(self, y_pred, y_true, others):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.others.append(others.detach().cpu())

    def save_records(self, path):
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        y_true = torch.cat(self.y_true, dim=0).numpy()
        if all(item is None for item in self.others):
            others = None
        else:
            others = torch.cat(self.others, dim=0).numpy()
        if path is not None:
            save_dict = dict(y_pred=y_pred, y_true=y_true)
            if others is not None:
                save_dict['others'] = others
            np.savez(path, **save_dict)
        return y_pred, y_true, others


def test_model(test_data_root, road_data, model_path, kfold=3, batch_size=128,
               test_fn='call', test_fn_args=None, metric_type='continuous',
               output_path=None):
    """
    验证、测试数据集的精度指标

    :param test_fn: 调用的测试函数：'call'：调用model(*x)；
                    'p_sample'：调用model.p_sample_loop(*x)，用于扩散模型
                    callable: 传入的函数，参数为test_fn(model, *x)
    :param metric_type：'discrete': 离散值，预测路段/区域
                        'continuous': 连续值，预测目的地xy坐标
    :param output_path: 将模型输出结果、真值和其他必要指标保存到npz文件，
                        用于后续分析
    """
    assert test_fn in ['call', 'p_sample'] or callable(test_fn)
    assert metric_type in ['discrete', 'continuous']
    test_fn_args = {} if test_fn_args is None else test_fn_args

    def test_fn_call(model, *args, **kwargs):
        return model(*args, **kwargs)

    def test_fn_psample(model, *args, **kwargs):
        return model.p_sample_loop_repeat(*args, **kwargs)

    model_path = Path(model_path)
    print(f'test {model_path.name}')
    test_data = TrajectoryData(test_data_root, road_data, kfold=kfold, other_cols=VAL_OTHERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.collate,
                             num_workers=0, pin_memory=True)
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    if test_fn in ['call', 'p_sample']:
        test_fn = {'call': test_fn_call, 'p_sample': test_fn_psample}[test_fn]
    if metric_type == 'discrete':
        val_fn1 = {'acc1': accuracy_, 'acc5': partial(accuracyK_, k=5),
                   'acc10': partial(accuracyK_, k=10), 'f1score': f1score_, 'mrr': mrr_}
        val_fn2 = {}
    elif metric_type == 'continuous':
        val_fn1 = {'mean_dist': mean_dist_, 'median_dist': MedianDist(),
                   'acc_500m': partial(accuracy_range_, distance=500),
                   'acc_1km': partial(accuracy_range_, distance=1000)}
        val_fn2 = {'mre': mre_, 'median_rel_err': MedianRelErr()}
    else:
        raise NotImplementedError
    val_fn = {**val_fn1, **val_fn2}

    ep_acc = []
    test_recorder = TestRecorder()
    with tqdm(test_loader, mininterval=TQDM_UPDATE_TIME) as pbar, \
            torch.inference_mode():
        for batch in pbar:
            batch = _to_device(batch, DEVICE)
            if len(batch) == 3:
                x, y, others = batch
            else:
                x, y = batch
                others = None
            pred = test_fn(model, *x, **test_fn_args)
            if isinstance(pred, tuple):
                pred = pred[0]
            test_recorder.add_batch(pred, y[0], others)
            if isinstance(model, DiffusionPredictor):
                pred = model.cal_dense_center(pred)
            batch_acc = {name: fn(pred, y[0]) for name, fn in val_fn1.items()}
            batch_acc.update({name: fn(pred, y[0], others[:, 0]) for name, fn in val_fn2.items()})
            ep_acc.append(tuple(batch_acc.values()))
            pbar.set_description(', '.join(f'{name}: {val:>7f}' for name, val in batch_acc.items()),
                                 refresh=False)
        ep_acc = torch.tensor(ep_acc).mean(dim=0)
    ep_acc = {name: _cls.median() if isinstance(_cls, BaseMedianFn) else ep_acc[i].item()
              for i, (name, _cls) in enumerate(val_fn.items())}
    print(', '.join(f'{name} = {val:>7f}' for name, val in ep_acc.items()))
    if output_path is not None:
        test_recorder.save_records(output_path)
    return pd.Series(ep_acc)


if __name__ == '__main__':
    encoder_name = 'transformer5_poiall3'
    model_name = 'diffusion3_all_feat'
    set_module_param(_DEVICE=DEVICE, _TQDM_UPDATE_TIME=60)

    # 训练基本模型f_phi(x)
    print('################', f'run {__file__} at {time.asctime()}',
          f'train model: {enocder_name}', sep='\n', flush=True)
    train_data = TrajectoryDataInMemAutoReg(TRAJ_ROOT / 'trajectories', TRAJ_ROOT / 'roads/road_input.csv',
                                            kfold=[0, 1, 2], n_jobs=12)
    val_data = TrajectoryDataInMem(TRAJ_ROOT / 'trajectories', TRAJ_ROOT / 'roads/road_input.csv',
                                   kfold=3, n_jobs=12)
    road_weight = np.load(TRAJ_ROOT / 'embedding/roademb_line_d32_neg10_ep500.npy')
    # poi_weight = np.load(TRAJ_ROOT / 'embedding/POIemb_500m_k10_rel.npz')
    base_model = TransfmPredictor(emb_dim=32, n_layer=4, n_head=4, max_len=train_data.seq_len,
                                  dropout=0.1, road_weight=torch.as_tensor(road_weight),
                                  y_num=4).to(DEVICE)
    train_model(train_data, base_model, batch_size=600, epoch=100, val_data=None, val_freq=10,
                lr=1e-3, train_mode='autoreg3_xy', output_path=TRAJ_ROOT / f'predict_model/{encoder_name}')

    # 训练扩散模型
    # base_model = torch.load(f'predict_model/{encoder_name}.pth', map_location=DEVICE)
    diff_model = DiffusionPredictor(2, base_model, emb_dim=32, diff_steps=100, schedule='linear',
                                    n_layer=4, n_head=4, dropout=0.1).to(DEVICE)
    train_model(train_data, diff_model, batch_size=600, epoch=100, val_data=val_data, val_freq=10,
                lr=1e-3, train_mode='diffusion_autoreg_xy',
                output_path=TRAJ_ROOT / f'predict_model/{model_name}')

    # 测试数据集精度
    test_model(TRAJ_ROOT / 'trajectories', TRAJ_ROOT / 'roads/road_input.csv',
               TRAJ_ROOT / f'predict_model/{model_name}.pth', batch_size=128,
               test_fn='p_sample', test_fn_args=dict(pred_y_index=1, repeat=32),
               metric_type='continuous',
               output_path=TRAJ_ROOT / f'predict_model/{model_name}_psample.npz')
