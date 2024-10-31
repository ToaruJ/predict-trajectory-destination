# 轨迹目的地预测模型：数据迭代器部分


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from pathlib import Path
import joblib as jl
from globalval import *


__all__ = ['TrajectoryData', 'TrajectoryDataInMem', 'TrajectoryDataInMemAutoReg',
           'TrajectoryDataStaticLen', 'TrajectoryDataSingleSample']


class TrajectoryData(IterableDataset):
    """
    轨迹数据集的迭代器，用于循环读取轨迹。内部自带epoch打乱样本顺序。
    默认batch_first = False, 即输出的序列维度是 (seq_len, bsize, emb_dim)。

    浮点数的归一化、缩放处理：
    经纬度(lon, lat)先投影至平面(EPSG:4526)后，等角映射到[-1, 1]（东西向）。
    出发时间(一天内的hour, minute, second) -> [0, 1)。
    POI数量：将[0, 数据集中出现的最大值] -> [0, 1)。
    路程耗时：将轨迹的路途用时转成hour单位：当前数据集范围为[0.034, 0.805]。
    速度：将km/h为单位的数值除以120，即[0, 120] -> [0, 1]。
    padding值：0值作为序列的padding值，原数据中序列点的one-hot值都+1。

    :param traj_root: 轨迹数据集的目录，里面有个`metadata.csv`记录所有轨迹的元数据
    :param road_data: 路段的空间、属性数据文件
    :param kfold: 需要使用的数据集代号。原始数据集内将轨迹每天5等分，切分出训练、验证、测试集
    :param min_seqlen: 对轨迹mask时，保留的最短轨迹长度
    :param seq_len: 输出的轨迹序列长度。若实际轨迹长度不足，则用-1来pad
    :param class_weight: 不同的目的地标签，是否拥有不同的权重，默认表示等权重。
        支持的选项有：'inv': 按照出现频率倒数关系，即 w = 1 / (1 + freq)；
                     'loginv': 按照出现频率的对数倒数关系，w = 1 / log(e + freq)；
    :param other_cols：batch输出的其他列。None时collate只输出(x, y)，指定other_cols输出(x, y, other_cols)
    """
    def __init__(self, traj_root, road_data, kfold, min_seqlen=10,
                 seq_len=240, class_weight=None, n_jobs=2, other_cols=None):
        assert class_weight in ['inv', 'loginv', None]

        self.traj_root = Path(traj_root)
        self.n_jobs = n_jobs
        self.min_seqlen = min_seqlen
        self.seq_len = seq_len
        self.kfold = {kfold} if isinstance(kfold, int) else set(kfold)
        self.meta = self._process_metadata()
        self.date_group = self.meta.groupby(['month', 'day'])
        self.date = list(self.date_group.groups.keys())
        self.roads = pd.read_csv(road_data, index_col='road1_id')
        self.num_roads = len(self.roads)
        self.class_weight = self._cal_class_weight(class_weight)
        self.other_cols = other_cols

    def _process_metadata(self):
        """读取轨迹元数据文件，并进行相应处理"""
        _meta = pd.read_csv(self.traj_root / 'metadata.csv', index_col='trace_id')
        _meta = _meta[_meta['k_fold'].isin(self.kfold)].copy()
        _meta['time_interval'] = (_meta['end_time'] - _meta['start_time']) / 3600
        _meta['start_t'] = _meta['start_h'] * 4 + _meta['start_m'] // 15
        return _meta

    def _cal_class_weight(self, class_weight):
        if class_weight is None:
            return None
        _bincnt = torch.bincount(torch.as_tensor(self.meta[INPUT_Y_INT[0]].values),
                                 minlength=self.num_roads)
        if class_weight == 'inv':
            return 1 / (1 + _bincnt.to(torch.float))
        elif class_weight == 'loginv':
            return 1 / torch.log(torch.e + _bincnt.to(torch.float))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.meta)

    def __iter__(self):
        return iter(self.generator())

    def generator(self):
        """生成单个样本：(轨迹ID, 轨迹序列的整数属性, 浮点数数属性, 轨迹前段的随机长度mask)"""
        with ProcessPoolExecutor(self.n_jobs) as pool:
            rand_dates = shuffle(self.date)
            q = Queue(self.n_jobs)
            for i in range(min(self.n_jobs, len(rand_dates))):
                q.put(pool.submit(self._read_one_file, rand_dates[i]))
            for i in range(len(rand_dates)):
                date, value = q.get().result()
                if i < len(rand_dates) - self.n_jobs:
                    q.put(pool.submit(self._read_one_file, rand_dates[i + self.n_jobs]))
                trace_ids, lines_int, lines_float = value
                rand_len = self._gen_seq_len(trace_ids)
                for trace_id, line_i, line_f, seq_len in \
                        shuffle(list(zip(trace_ids, lines_int, lines_float, rand_len))):
                    yield trace_id, \
                          torch.tensor(line_i[:seq_len, :], dtype=torch.long), \
                          torch.tensor(line_f[:seq_len, :], dtype=torch.float)

    def _read_one_file(self, date):
        """
        读取一个轨迹文件（一天的轨迹），并进行初步处理
        数据merge时一定要`how='left'`：目前发现的特性是：
        merge时若发现左表有相同的值，则inner merge后会聚在一起，而left merge会保持左表的原来顺序。
        """
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()
        file['tm'] = file['tm'].astype(np.float_)
        file['tm'] = file.groupby('trace_id')['tm'].transform(lambda val: val - val.min())
        file = file.merge(self.roads, how='left', left_on='road1_id', right_index=True)
        values = tuple(zip(*((id_, traj[INPUT_INT].values, traj[INPUT_FLOAT].values)
                             for id_, traj in file.groupby('trace_id'))))
        return date, values

    def _gen_seq_len(self, trace_ids):
        """生成一定长度的子序列，基类实现的是随机长度，子类中会固定长度"""
        return np.random.randint(self.min_seqlen,
                                 self.meta.loc[np.array(trace_ids), 'num_record'])

    def collate(self, batch):
        """
        由多个样本生成batch，用于代替Dataloader的collate_fn方法。
        包括整合metadata，轨迹序列的padding，mask补充。
        注：输出的数据中，序列的pad整数部分由-1填充。

        :return: (轨迹元数据的整数信息, 浮点数信息, 轨迹每个记录点的整数数据, 浮点数数据, 轨迹mask), 标签。
            数据size： ((bsize, d1), (bsize, d2), (seq_len, bsize, d3), (seq_len, bsize, d4),
                       (bsize, seq_len)), ((bsize, d5), (bsize, d6))
        """
        trace_id, seq_int, seq_float = zip(*batch)
        meta_int = self.meta.loc[trace_id, INPUT_META_INT].values
        meta_int = torch.as_tensor(meta_int, dtype=torch.long)
        meta_float = self.meta.loc[trace_id, INPUT_META_FLOAT].values
        meta_float = torch.as_tensor(meta_float, dtype=torch.float)
        seq_int = pad_sequence(seq_int, padding_value=-1)
        seq_float = pad_sequence(seq_float, padding_value=0)
        seq_mask = (seq_int[:, :, 0] == -1).mT
        # y_int = self.meta.loc[trace_id, INPUT_Y_INT]
        # y_int = torch.as_tensor(y_int.values, dtype=torch.long).squeeze(-1)
        y_float = self.meta.loc[trace_id, INPUT_Y_FLOAT]
        y_float = torch.as_tensor(y_float.values, dtype=torch.float)
        y_xycoor, y_time = y_float.split([2, len(INPUT_Y_FLOAT) - 2], dim=-1)
        if self.other_cols:
            others = self.meta.loc[trace_id, self.other_cols].values
            others = torch.as_tensor(others, dtype=torch.float)
            trace_len = torch.count_nonzero(~seq_mask, dim=1)
            others = torch.cat([others, torch.tensor(trace_id, dtype=torch.float).unsqueeze(-1),
                                trace_len.to(torch.float).unsqueeze(-1)], dim=1)
            return (meta_int, meta_float, seq_int, seq_float, seq_mask), (y_xycoor, y_time), others
        return (meta_int, meta_float, seq_int, seq_float, seq_mask), (y_xycoor, y_time)


class TrajectoryDataInMem(TrajectoryData):
    """
    `TrajectoryData`类的全内存版本，将整个数据集放入内存（服务器可用）。

    :param traj_root: 轨迹数据集的目录，里面有个`metadata.csv`记录所有轨迹的元数据
    :param road_data: 路段的空间、属性数据文件
    :param kfold: 需要使用的数据集代号。原始数据集内将轨迹每天5等分，切分出训练、验证、测试集
    :param min_seqlen: 对轨迹mask时，保留的最短轨迹长度
    :param seq_len: 输出的轨迹序列长度。若实际轨迹长度不足，则用-1来pad
    :param class_weight: 不同的目的地标签，是否拥有不同的权重，默认表示等权重。
        支持的选项有：'inv': 按照出现频率倒数关系，即 w = 1 / (1 + freq)；
                     'loginv': 按照出现频率的对数倒数关系，w = 1 / log(e + freq)；
    """

    def __init__(self, traj_root, road_data, kfold, min_seqlen=10,
                 seq_len=240, class_weight=None, n_jobs=8, other_cols=None):
        super(TrajectoryDataInMem, self).__init__(
            traj_root, road_data, kfold, min_seqlen, seq_len, class_weight, n_jobs, other_cols)
        self.files = None
        self.load_data()

    def load_data(self):
        print(f'load {len(self.date)} trajectory files', flush=True)
        with jl.Parallel(self.n_jobs, verbose=5, backend='multiprocessing') as paral:
            files = paral(jl.delayed(self._read_one_file)(date)
                          for date in self.date)
        self.files = dict(files)

    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int, lines_float = self.files[date]
            rand_len = self._gen_seq_len(trace_ids)
            for trace_id, line_i, line_f, seq_len in \
                    shuffle(list(zip(trace_ids, lines_int, lines_float, rand_len))):
                yield trace_id, \
                      torch.tensor(line_i[:seq_len, :], dtype=torch.long), \
                      torch.tensor(line_f[:seq_len, :], dtype=torch.float)


class TrajectoryDataInMemAutoReg(TrajectoryDataInMem):
    """
    `TrajectoryDataInMem`类的改版，适用于自回归训练
    """
    # 尝试按照类RNN方式输入数据（对于每个轨迹点的Transformer输出，都预测一次目的地）
    # TODO 这里已经是输出整条序列
    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int, lines_float = self.files[date]
            for trace_id, line_i, line_f in \
                    shuffle(list(zip(trace_ids, lines_int, lines_float))):
                yield trace_id, \
                      torch.tensor(line_i, dtype=torch.long), \
                      torch.tensor(line_f, dtype=torch.float)


class TrajectoryDataStaticLen(TrajectoryData):
    """
    固定长度、或者固定输入长度比例的轨迹数据集。轨迹数据不再是随机长度的序列。

    :param len_ratio: 输入轨迹的长度比例，(0, 1)之间的值，
                      此时每条轨迹都截取相同长度比例的前段，输入模型中
    :param seq_len: seq_len > 0时，表示输入轨迹的长度（记录点数），所有轨迹都被截短至固定长度。
                    seq_len < 0时，表示去除轨迹末尾部分的长度，所有轨迹的最后abs(seq_len)个点未知。
                    本身长度不足的轨迹会被丢弃。`len_ratio`和`seq_len`只能输入一个
    """
    def __init__(self, traj_root, road_data, kfold, len_ratio=None,
                 seq_len=None, class_weight=None, n_jobs=2, other_cols=None):
        assert len_ratio is not None or seq_len is not None
        assert not (len_ratio is not None and seq_len is not None)
        if len_ratio is not None:
            assert 0 < len_ratio < 1

        super(TrajectoryDataStaticLen, self).__init__(
            traj_root, road_data, kfold, seq_len=seq_len,
            class_weight=class_weight, n_jobs=n_jobs, other_cols=other_cols)
        self.len_ratio = len_ratio

    def _process_metadata(self):
        _meta = super(TrajectoryDataStaticLen, self)._process_metadata()
        if self.seq_len is not None:
            return _meta[_meta['num_record'] > abs(self.seq_len)].copy()
        return _meta

    def _gen_seq_len(self, trace_ids):
        """重载了生成子序列长度的方法，按照固定长度/固定长度比例生成预测序列"""
        if self.len_ratio is not None:
            len_ = self.meta.loc[np.array(trace_ids), 'num_record'] * self.len_ratio
            return np.round(len_).astype(np.int_)
        else:
            return np.full((len(trace_ids),), self.seq_len)


class TrajectoryDataSingleSample(TrajectoryDataStaticLen):
    """
    可以指定轨迹ID、已知轨迹长度信息的数据产生器，遍历整个数据集的性能很差。用于模型输出示例绘图。
    """
    def __init__(self, traj_root, road_data, kfold=None, len_ratio=None,
                 seq_len=None, class_weight=None):
        if kfold is None:
            kfold = set(range(5))
        super(TrajectoryDataSingleSample, self).__init__(
            traj_root, road_data, kfold, len_ratio=len_ratio,
            seq_len=seq_len, class_weight=class_weight, n_jobs=1)
        self.file_dict = {}

    def __getitem__(self, item):
        """:param item: trace_id字段"""
        assert isinstance(item, int)
        month_day = tuple(self.meta.loc[item, ['month', 'day']].astype(int))
        if month_day not in self.file_dict:
            self.file_dict = {month_day: self._read_one_file(month_day)[1]}
        file_data = self.file_dict[month_day]
        seq_len = self._gen_seq_len(item)
        for trace_id, line_i, line_f in list(zip(*file_data)):
            if trace_id != item:
                continue
            tensor_i = torch.tensor(line_i[:seq_len, :], dtype=torch.long)
            tensor_f = torch.tensor(line_f[:seq_len, :], dtype=torch.float)
            return self.collate([(trace_id, tensor_i, tensor_f)])
