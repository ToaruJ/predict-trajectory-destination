import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from globalval import POI_dict, RESEARCH_BOUND
from predict_dataset import TrajectoryDataInMem, shuffle


INPUT_INT = ['poi']
INPUT_FLOAT = ['x', 'y', 'speed', 'dis', 'angle']
INPUT_META_INT = ['weekday', 'start_t']
INPUT_META_FLOAT = [*POI_dict.keys()]
INPUT_Y = ['dest_x', 'dest_y']
RESEARCH_CENTER = np.array([RESEARCH_BOUND[2] + RESEARCH_BOUND[0],
                            RESEARCH_BOUND[3] + RESEARCH_BOUND[1]]) / 2
RESOLUTION = 2 / max(RESEARCH_BOUND[2] - RESEARCH_BOUND[0],
                     RESEARCH_BOUND[3] - RESEARCH_BOUND[1])


class Dataset(TrajectoryDataInMem):
    def _process_metadata(self):
        _meta = super(Dataset, self)._process_metadata()
        _meta['start_t'] = _meta['start_h'] * 2 + _meta['start_m'] // 30
        return _meta

    def _read_one_file(self, date):
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()
        # 文件中speed的单位是[0, 120]km/s -> [0, 1]，需变回m/s单位
        file['speed'] = file['speed'] * 120 / 3.6
        # 距离出发转成km单位
        groupby = file.groupby('trace_id', as_index=False, group_keys=False)
        file['dis'] = groupby.apply(lambda traj: np.sqrt(
            np.square(traj[['x', 'y']].diff().fillna(0)).sum(axis=1)).cumsum()
                                                   / RESOLUTION / 1000)
        file['angle'] = groupby.apply(self._calc_angle)
        del groupby
        file = file.merge(self.roads, how='left', left_on='road1_id', right_index=True)
        file['poi'] = np.argmax(file[POI_dict.keys()].values, axis=1)
        values = tuple(zip(*((id_, traj[INPUT_INT].values, traj[INPUT_FLOAT].values)
                             for id_, traj in file.groupby('trace_id'))))
        return date, values

    @classmethod
    def _calc_angle(cls, traj):
        diff = np.diff(traj[['x', 'y']], axis=0)
        diff_c = diff[:, 0] + diff[:, 1] * 1j
        result = np.zeros((traj.shape[0],))
        result[1:-1] = np.abs(np.angle(diff_c[1:] / diff_c[:-1]))
        return pd.Series(result, index=traj.index)

    def collate(self, batch):
        trace_id, seq_int, seq_float = zip(*batch)
        meta_int = self.meta.loc[trace_id, INPUT_META_INT].values
        meta_int = torch.as_tensor(meta_int, dtype=torch.long)
        meta_float = self.meta.loc[trace_id, INPUT_META_FLOAT].values
        meta_float = torch.as_tensor(meta_float, dtype=torch.float)
        seq_int = pad_sequence(seq_int, padding_value=-1)
        seq_float = pad_sequence(seq_float, padding_value=0)
        seq_mask = (seq_int[:, :, 0] == -1).mT
        y_float = self.meta.loc[trace_id, INPUT_Y]
        y_float = torch.as_tensor(y_float.values, dtype=torch.float)
        if self.other_cols:
            others = self.meta.loc[trace_id, self.other_cols].values
            others = torch.as_tensor(others, dtype=torch.float)
            return (meta_int, meta_float, seq_int, seq_float, seq_mask), (y_float, None), others
        return (meta_int, meta_float, seq_int, seq_float, seq_mask), (y_float, None)
