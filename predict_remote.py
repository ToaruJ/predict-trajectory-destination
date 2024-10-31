# 用于在Linux服务器上运行程序，是`predict_trainval.py`在服务器上执行的代码


import time
import numpy as np
import torch
from predict_dataset import TrajectoryDataInMem, TrajectoryDataInMemAutoReg
from predict_nn import TransfmPredictor, DiffusionPredictor
from predict_trainval import train_model, set_module_param, test_model
from globalval import *


DEVICE = torch.device("cuda:3")


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
