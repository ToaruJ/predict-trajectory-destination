import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from predict_dataset import TrajectoryDataInMem, shuffle


INPUT_INT = ['road1_id']
INPUT_Y = ['dest_x', 'dest_y']


class Dataset4TNV(TrajectoryDataInMem):
    def __init__(self, traj_root, road_data, kfold, window_size=5,
                 min_seqlen=10, seq_len=240, n_jobs=8):
        super(Dataset4TNV, self).__init__(traj_root, road_data, kfold, min_seqlen,
                                          seq_len, n_jobs=n_jobs)
        self.window_size = window_size
        self._len = len(self.meta) * window_size * 2

    def _read_one_file(self, date):
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()
        file = file.merge(self.roads, how='left', left_on='road1_id', right_index=True)
        values = []
        for id_, traj in file.groupby('trace_id'):
            # 去除重复出现的路段
            select = np.ones((traj.shape[0],), np.bool_)
            select[1:] = traj['road1_id'].iloc[1:].values != traj['road1_id'].iloc[0:-1].values
            traj = traj[select]
            values.append(traj[INPUT_INT].values)
        return date, values

    def __len__(self):
        return self._len

    def generator(self):
        for date in shuffle(self.date):
            for line_i in shuffle(self.files[date]):
                for i in range(1, min(self.window_size + 1, line_i.shape[0])):
                    p1 = torch.as_tensor(line_i[i:, 0], dtype=torch.long)
                    p2 = torch.as_tensor(line_i[:-i, 0], dtype=torch.long)
                    yield p1, p2
                    yield p2, p1

    def collate(self, batch):
        x, y = tuple(zip(*batch))
        return torch.cat(x, dim=0), torch.cat(y, dim=0)


class DatasetTrain(TrajectoryDataInMem):
    def load_data(self):
        super(DatasetTrain, self).load_data()
        real_ids = [id_ for ids, _ in self.files.values() for id_ in ids]
        self.meta = self.meta.loc[real_ids].copy()
        num_record = [traj.shape[0] for _, trajs in self.files.values() for traj in trajs]
        self.meta.loc[real_ids, 'num_record'] = num_record

    def _read_one_file(self, date):
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()
        file = file.merge(self.roads, how='left', left_on='road1_id', right_index=True)
        values = []
        for id_, traj in file.groupby('trace_id'):
            # 去除重复出现的路段
            select = np.ones((traj.shape[0],), np.bool_)
            select[1:] = traj['road1_id'].iloc[1:].values != traj['road1_id'].iloc[0:-1].values
            traj = traj[select]
            if traj.shape[0] > self.min_seqlen:
                values.append((id_, traj[INPUT_INT].values))
        values = tuple(zip(*((id_, val) for id_, val in values)))
        return date, values

    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int = self.files[date]
            for trace_id, line_i in shuffle(list(zip(trace_ids, lines_int))):
                yield trace_id, torch.tensor(line_i, dtype=torch.long)

    def collate(self, batch):
        trace_id, seq_int = zip(*batch)
        seq_int = pad_sequence(seq_int, padding_value=-1)
        seq_mask = torch.all(seq_int == -1, dim=-1).mT
        y_float = self.meta.loc[trace_id, INPUT_Y]
        y_float = torch.as_tensor(y_float.values, dtype=torch.float)
        if self.other_cols:
            others = self.meta.loc[trace_id, self.other_cols].values
            others = torch.as_tensor(others, dtype=torch.float)
            return (None, None, seq_int, None, seq_mask), (y_float, None), others
        return (None, None, seq_int, None, seq_mask), (y_float, None)


class DatasetTest(DatasetTrain):
    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int = self.files[date]
            rand_len = self._gen_seq_len(trace_ids)
            for trace_id, line_i, seq_len in \
                    shuffle(list(zip(trace_ids, lines_int, rand_len))):
                yield trace_id, torch.tensor(line_i[:seq_len, :], dtype=torch.long)