import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
# from globalval import TRAJ_ROOT
from .model import TNV
from .dataset import Dataset4TNV
from predict_trainval import _to_device


assert torch.cuda.is_available()
DEVICE = torch.device("cuda:1")
TQDM_UPDATE_TIME = 60


def TNVtrain(train_data, model, batch_size=1024, epoch=100, lr=1e-3,
             output_path=None):
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate,
                              pin_memory=True)
    history = []
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for i in range(epoch):
        ep_loss = []
        with tqdm(train_loader, mininterval=TQDM_UPDATE_TIME) as pbar:
            for batch in pbar:
                x, y = _to_device(batch, DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)
                pbar.set_description(f'epoch {i:>3d}, batch loss: {loss.item():>7f}',
                                     refresh=False)
                ep_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        ep_loss = torch.tensor(ep_loss).mean(dim=0)
        ep_str = [f'loss = {ep_loss.item():>7f}']
        ep_dict = {'epoch': i, 'loss': ep_loss.item()}
        print(f'epoch {i:>3d}:', ', '.join(ep_str), flush=True)
        history.append(ep_dict)

    # 保存训练结果
    if output_path is not None:
        output_path = Path(output_path)
        history = pd.DataFrame(history)
        history.to_csv(output_path.with_suffix('.csv'), index=False)
        np.savez(Path(output_path).with_suffix('.npz'),
                 emb_center=model.embedding_center().cpu().numpy(),
                 emb_context=model.embedding_context().cpu().numpy())
    return model, history


if __name__ == '__main__':
    TRAJ_ROOT = Path('.')
    road_df = pd.read_csv(TRAJ_ROOT / 'roads/road_input.csv')
    model = TNV(road_df.shape[0]).to(DEVICE)
    train_data = Dataset4TNV(TRAJ_ROOT / 'trajectories',
                             TRAJ_ROOT / 'roads/road_input.csv',
                             kfold=[0, 1, 2], window_size=5, n_jobs=12)
    TNVtrain(train_data, model, batch_size=128, epoch=50,
             output_path=TRAJ_ROOT / 'embedding/baseline_tnv2019')
