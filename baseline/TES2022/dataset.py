import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from predict_dataset import TrajectoryDataInMem, shuffle

INPUT_META = ['start_t']
INPUT_INT = ['xy_id']
INPUT_Y = ['dest_x', 'dest_y']


class DatasetTest(TrajectoryDataInMem):
    NUM_CONTEXT = len(INPUT_META)

    def __init__(self, traj_root, road_data, kfold, min_seqlen=10,
                 seq_len=240, class_weight=None, n_jobs=8, other_cols=None,
                 num_grid_axis=50):
        self.num_axis_grid = num_grid_axis
        super(DatasetTest, self).__init__(traj_root, road_data, kfold, min_seqlen,
                                          seq_len, class_weight, n_jobs, other_cols)

    def _read_one_file(self, date):
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()

        xy = (file[['x', 'y']] + 1) * self.num_axis_grid / 2
        xy = xy.astype(int).clip(0, self.num_axis_grid - 1)
        file['xy_id'] = xy['y'] * self.num_axis_grid + xy['x']
        values = tuple(zip(*((id_, traj[INPUT_INT].values)
                             for id_, traj in file.groupby('trace_id'))))
        return date, values

    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int = self.files[date]
            rand_len = self._gen_seq_len(trace_ids)
            for trace_id, line_i, seq_len in \
                    shuffle(list(zip(trace_ids, lines_int, rand_len))):
                yield trace_id, torch.tensor(line_i[:seq_len, :], dtype=torch.long)

    def collate(self, batch):
        trace_id, seq_int = zip(*batch)
        meta_int = self.meta.loc[trace_id, INPUT_META].values
        meta_int = torch.as_tensor(meta_int, dtype=torch.long)
        seq_int = pad_sequence(seq_int, padding_value=-1)
        seq_mask = torch.all(seq_int == -1, dim=-1).mT
        y_float = self.meta.loc[trace_id, INPUT_Y]
        y_float = torch.as_tensor(y_float.values, dtype=torch.float)
        if self.other_cols:
            others = self.meta.loc[trace_id, self.other_cols].values
            others = torch.as_tensor(others, dtype=torch.float)
            return (meta_int, None, seq_int, None, seq_mask), (y_float, None), others
        return (meta_int, None, seq_int, None, seq_mask), (y_float, None)


class DatasetTrain(DatasetTest):
    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_int = self.files[date]
            for trace_id, line_i in shuffle(list(zip(trace_ids, lines_int))):
                yield trace_id, torch.tensor(line_i, dtype=torch.long)
