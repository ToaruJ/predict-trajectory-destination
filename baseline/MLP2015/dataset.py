import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from predict_dataset import TrajectoryDataInMem, shuffle

INPUT_META = ['week', 'weekday', 'start_t']
INPUT_FLOAT = ['x', 'y']
INPUT_Y = ['dest_x', 'dest_y']


class Dataset(TrajectoryDataInMem):
    def _process_metadata(self):
        _meta = super(Dataset, self)._process_metadata()
        _meta['week'] = _meta['day'] // 7
        return _meta

    def _read_one_file(self, date):
        """
        读取一个轨迹文件（一天的轨迹），并进行初步处理
        数据merge时一定要`how='left'`：目前发现的特性是：
        merge时若发现左表有相同的值，则inner merge后会聚在一起，而left merge会保持左表的原来顺序。
        """
        filename = f'2017{date[0]:02d}{date[1]:02d}.csv'
        file = pd.read_csv(self.traj_root / filename)
        file = file[file['trace_id'].isin(self.date_group.groups[date])].copy()
        file = file.merge(self.roads, how='left', left_on='road1_id', right_index=True)
        values = tuple(zip(*((id_, traj[INPUT_FLOAT].values)
                             for id_, traj in file.groupby('trace_id'))))
        return date, values

    def generator(self):
        for date in shuffle(self.date):
            trace_ids, lines_float = self.files[date]
            rand_len = self._gen_seq_len(trace_ids)
            for trace_id, line_f, seq_len in \
                    shuffle(list(zip(trace_ids, lines_float, rand_len))):
                yield trace_id, torch.tensor(line_f[:seq_len, :], dtype=torch.float)

    def collate(self, batch):
        """
        由多个样本生成batch，用于代替Dataloader的collate_fn方法。
        包括整合metadata，轨迹序列的padding，mask补充。
        注：输出的数据中，序列的pad整数部分由-1填充。

        :return: (轨迹元数据的整数信息, 浮点数信息, 轨迹每个记录点的整数数据, 浮点数数据, 轨迹mask), 标签。
            数据size： ((bsize, d1), (bsize, d2), (seq_len, bsize, d3), (seq_len, bsize, d4),
                       (bsize, seq_len)), ((bsize, d5), (bsize, d6))
        """
        trace_id, seq_float = zip(*batch)
        meta_int = self.meta.loc[trace_id, INPUT_META].values
        meta_int = torch.as_tensor(meta_int, dtype=torch.long)
        seq_float = pad_sequence(seq_float, padding_value=0)
        seq_mask = torch.all(seq_float == 0, dim=-1).mT
        y_float = self.meta.loc[trace_id, INPUT_Y]
        y_float = torch.as_tensor(y_float.values, dtype=torch.float)
        if self.other_cols:
            others = self.meta.loc[trace_id, self.other_cols].values
            others = torch.as_tensor(others, dtype=torch.float)
            return (meta_int, None, None, seq_float, seq_mask), (y_float,), others
        return (meta_int, None, None, seq_float, seq_mask), (y_float,)
