# 实验用到的全局变量：包括文件路径、表格的字段名和对应意义、标识变量的映射……

from pathlib import Path
import datetime as dt

__all__ = ['TRAJ_ROOT', 'ROAD_SEG_ROOT', 'traj_header', 'correspondence_header',
           'POI_dict', 'ROADCLASS_map', 'FORMWAY_map', 'LINKTYPE_map',
           'PROJ_EPSG', 'RESEARCH_BOUND', 'VACATION',
           'INPUT_INT', 'INPUT_FLOAT', 'INPUT_META_INT',
           'INPUT_META_FLOAT', 'INPUT_Y_INT', 'INPUT_Y_FLOAT', 'VAL_OTHERS']

# 文件存储路径
TRAJ_ROOT = Path('..')
ROAD_SEG_ROOT = TRAJ_ROOT / '北京和深圳两个区的POI及路网数据/17Q4/深圳/'

# 轨迹、路网文件中的字段名
traj_header = ['trace_id', 'tm', 'id', 'speed', 'dir', 'x', 'y', 'link_id']
correspondence_header = ['link_id', 'mesh_code', 'road_id', 'dir']

# 研究中使用到的POI类型，用regexp表示。
# 从高德POI类别代码->到自定义大类的映射关系
POI_dict = {
    'auto_service': ['0[1-4]\\d{4}'],
    'food': ['05\\d{4}'],
    'shopping': ['06\\d{4}'],
    'daily_life': ['070[4-69]\\d{2}', '07[12]\\d{3}', '1601\\d{2}'],
    'recreation': ['08\\d{4}'],
    'medical': ['09\\d{4}'],
    'tourist': ['11\\d{4}'],
    'residence': ['10\\d{4}', '1203\\d{2}'],
    'government': ['13\\d{4}'],
    'cultural': ['14\\d{4}'],
    'transport': ['150[0-4]\\d{2}', '151000'],
    'enterprise': ['070[78]\\d{2}', '120[12]\\d{2}', '160[4-6]\\d{2}', '17\\d{4}']
}

# 路网属性到one-hot的映射
# 道路等级（高速、主干道、支路...）
ROADCLASS_map = {item: i for i, item in enumerate(
    [41000, 42000, 43000, 44000, 45000, 47000, 52000, 53000])}

# 道路的构成（上下行分离、辅路、转弯车道...）
FORMWAY_map = {item: i for i, item in enumerate(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])}

# 道路属性（普通道路、隧道、桥）
LINKTYPE_map = {item: i for i, item in enumerate([0, 2, 3, 4])}

# 研究区要计算距离时，默认的投影方式：CGCS2000 3度 Gauss-Kruger在深圳的位置（UTM 6度刚好跨区）
PROJ_EPSG = 'epsg:4526'

# 研究区的范围(minx, miny, maxx, maxy)（epsg:4526投影下的坐标），将该范围映射到模型的[-1, 1]
# 该映射是保持长宽比的，由于研究区东西向比南北向宽，因此是东西向保证[-1, 1]
RESEARCH_BOUND = (38484458, 2482940, 38510778, 2504753)

# 研究时间范围内的节假日标识
VACATION = [dt.date(2017, 10, i) for i in range(1, 9)] \
           + [dt.date(2017, 12, i) for i in range(30, 32)]
VACATION = set(VACATION)

# 输入到模型中的字段：轨迹中每个点含有的属性
INPUT_INT = ['road1_id', 'road_class', 'direction', 'form_way', 'link_type']
# INPUT_FLOAT = ['x', 'y', 'speed', 'tm']
INPUT_FLOAT = ['x', 'y', 'speed', 'numPOI', 'tm', *POI_dict.keys()]
# 输入到模型中的字段：轨迹的global属性，放在轨迹的开头位置
INPUT_META_INT = ['weekday', 'is_vacation', 'start_t']
# INPUT_META_FLOAT = ['start_t']
INPUT_META_FLOAT = ['numPOI', *POI_dict.keys()]
# INPUT_META_FLOAT = []
# 输入到模型中的字段：待预测标签
INPUT_Y_INT = ['dest_road']
INPUT_Y_FLOAT = ['dest_x', 'dest_y', 'time_interval']
# INPUT_Y_FLOAT = []
VAL_OTHERS = ['trip_len']
