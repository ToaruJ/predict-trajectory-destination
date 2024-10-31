# 数据预处理，使用高德深圳的数据。

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
from scipy.spatial.distance import cdist
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.utils import shuffle
from pathlib import Path
import joblib as jl
import time
import datetime as dt
import gzip
from tqdm import tqdm
import re
from globalval import *


# 路网文件的字段：['Mesh_Code', 'ROAD_ID', 'NAME_CHN', 'FNODE', 'TNODE', 'LENGTH',
#   'ROAD_CLASS', 'DIRECTION', 'AD_CODE', 'FORM_WAY', 'LINK_TYPE', 'geometry']
# 不同城区的correspondence文件中，'link_id'字段不会重复
# 轨迹数量：国庆节：福田106513，南山115289。周中：福田110065，南山118067。差异没那么大
# 如果限制两条记录间隔<5min，则剩下80k条左右，而限制<1min则只剩下40k条
# 出行用时统计：最短1:43，最长49:18。最长的轨迹有548个点（未清除重复路段）
#   目前的清理结果：最长用时35:23
# 要在深圳计算投影距离，可以用EPSG:4526，即CGCS2000 3度 Gauss-Kruger在深圳的位置（UTM 6度刚好跨区）
# 发现POI密度差异很大，比如商贸区两个POI点之间可能 < 10m，而稀疏的区域最邻近POI可达几十米
#   结合Place2vec的实验（Yan et al. (2017)），单独使用距离倒数权重的效果很差，甚至差于KNN+等权重，
#   猜测可能就是POI密度差异导致的


def geopoint_to_mp1(points, source_crs='epsg:4326'):
    """
    将一个坐标系下的点，根据研究区范围映射到[-1, 1]，
    若是epsg4526坐标的点，则直接线性等角变换，否则先投影到epsg4526。
    输入的points是shape = (N, 2)的numpy点集，点的坐标顺序是<x, y>或者<lon, lat>

    :param source_crs: 输入点`points`的坐标系
    """
    center = np.array([RESEARCH_BOUND[2] + RESEARCH_BOUND[0],
                       RESEARCH_BOUND[3] + RESEARCH_BOUND[1]]) / 2
    resolution = 2 / max(RESEARCH_BOUND[2] - RESEARCH_BOUND[0],
                         RESEARCH_BOUND[3] - RESEARCH_BOUND[1])
    center = center[None, :]
    # 可能需要的投影变换
    if source_crs != PROJ_EPSG:
        _p = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1]})
        _p = _p.apply(lambda l: geo.Point(l['x'], l['y']), axis=1)
        _p = gpd.GeoSeries(_p, crs=source_crs)
        _p = _p.to_crs(PROJ_EPSG)
        points = np.stack([_p.x, _p.y], axis=1)
    return (points - center) * resolution


def mp1_to_geopoint(points):
    """将[-1, 1]范围的坐标点，映射回epsg:4526坐标系。是`geopoint_to_mp1`的逆变换"""
    center = np.array([RESEARCH_BOUND[2] + RESEARCH_BOUND[0],
                       RESEARCH_BOUND[3] + RESEARCH_BOUND[1]]) / 2
    resolution = 2 / max(RESEARCH_BOUND[2] - RESEARCH_BOUND[0],
                         RESEARCH_BOUND[3] - RESEARCH_BOUND[1])
    center = center[None, :]
    return points / resolution + center


def merge_district_roadPOI():
    """合并南山区和福田区的路网和POI数据"""
    corresp = []
    shps = []
    pois = []
    for code, region, pinyin in [('04', '福田', 'ft'), ('05', '南山', 'ns')]:
        roads1 = pd.read_csv(ROAD_SEG_ROOT / f'{region}区/correspondence_{pinyin}_17q4.txt', header=None)
        roads1.columns = correspondence_header
        roads2 = gpd.read_file(ROAD_SEG_ROOT / f'{region}区/RoadSegment_17q4.TAB')
        poi = gpd.read_file(ROAD_SEG_ROOT / f'{region}区/POI_17q4.TAB')
        corresp.append(roads1)
        shps.append(roads2)
        pois.append(poi)
    corresp = pd.concat(corresp)
    shps = gpd.GeoDataFrame(pd.concat(shps), crs=shps[0].crs)
    pois = gpd.GeoDataFrame(pd.concat(pois), crs=pois[0].crs)
    corresp.to_csv(TRAJ_ROOT / 'roads/correspondence.csv', index=False)
    shps.to_file(TRAJ_ROOT / 'roads/road_network.shp')
    pois.to_file(TRAJ_ROOT / 'roads/poi.shp')
    return corresp, shps, pois


def clean_road1(shpfile, correspond_path):
    """清理道路数据1：去除correspond中没记录的几何体，以及小路"""
    shp = gpd.read_file(shpfile)
    correspond = pd.read_csv(correspond_path)
    shp = shp[(shp['ROAD_CLASS'] != 49) & (shp['ROAD_CLASS'] != 54000)]
    shp_id = shp.apply(lambda l: (l['Mesh_Code'], l['ROAD_ID']), axis=1)
    corresp_id = correspond.apply(lambda l: (l['mesh_code'], l['road_id']), axis=1)
    shp = shp[shp_id.isin(corresp_id)]
    shp_id = shp.apply(lambda l: (l['Mesh_Code'], l['ROAD_ID']), axis=1)
    correspond = correspond[corresp_id.isin(shp_id)]
    shp['road1_id'] = np.arange(len(shp))
    shp.to_file(Path(shpfile).with_name('road_1.shp'))
    correspond.to_csv(Path(shpfile).with_name('correspondence_1.csv'), index=False)
    return shp, correspond


def generate_line(frame, latname, lonname):
    """
    生成一条折线轨迹，轨迹以一个frame表示，每行表示一个点。
    """
    # 目前的问题：1. 有些轨迹记录其实明显不是单条，而是在空间上多段的。
    # 但是也有过隧道导致信号中断的，如何区分？注：南山区东滨隧道用时4min左右，
    x, y = list(frame[lonname]), list(frame[latname])
    return geo.LineString(zip(x, y))


def _clean_line1(frame, correspond, min_p=10, max_p=240, max_interval=60, min_time=300,
                 avail_speed_ratio=0.8, road_exist_ratio=0.8, impute_speed=True,
                 remove_repeat=False):
    """
    筛选折线轨迹。是否清理折线轨迹中的重复路段点？
    轨迹以一个frame表示，路段ID存储于correspondence中，每行表示一个点。
    未达到要求返回空表。
    注：轨迹里面可能混有走路的，仅取出驾车轨迹

    :param min_p: 最少的记录点数
    :param max_p: 最多的记录点数（过长的路径可能是绕路观光出行）
    :param max_interval: 两个轨迹点之间的最长时间间隔(s)
    :param min_time: 轨迹的最短总时长(s)
    :param avail_speed_ratio: 有效定位点（`speed`值 != 255）的比例，小于该比例则去除
    :param road_exist_ratio: 匹配到主要道路的轨迹点（该道路存在于`correspond`中）比例，小于该比例表明该轨迹不走主干路
    :param impute_speed: 插值无效的速度值（`speed` == 255），使用KNNImputer
    :param remove_repeat: 去除重复路段，仅保留进入该路段、即将离开该路段的记录，去除中间数据点
    """
    frame = frame.sort_values('tm')
    if len(frame) < min_p or np.max(np.diff(frame['tm'])) > max_interval \
            or np.max(frame['tm']) - np.min(frame['tm']) < min_time:
        return pd.DataFrame(columns=frame.columns)
    # 最高时速设为高速路120+定位误差or超速20，无法计算速度时会赋值255
    # 清除有效定位点比例过低的轨迹
    if np.count_nonzero(frame['speed'] != 255) / len(frame) < avail_speed_ratio:
        return pd.DataFrame(columns=frame.columns)
    # 清除走路、自行车的轨迹。认为驾车轨迹至少能达到20km/h的速度
    if frame.loc[frame['speed'] != 255, 'speed'].max() < 20:
        return pd.DataFrame(columns=frame.columns)
    # 给无效的速度值插值
    if impute_speed:
        data = frame[['tm', 'speed', 'x', 'y']].values.copy()
        data[:, 0] = (data[:, 0] - data[:, 0].min()) / data[:, 0].max()
        data[:, 1] = np.where(data[:, 1] != 255, data[:, 1], np.nan)
        data = KNNImputer(n_neighbors=4).fit_transform(data)
        frame['speed'] = data[:, 1]
    # 是否清理重复路段？
    if remove_repeat:
        select = np.ones((frame.shape[0],), np.bool_)
        select[1:-1] = (frame['link_id'].iloc[1:-1].values != frame['link_id'].iloc[0:-2].values) | \
                       (frame['link_id'].iloc[1:-1].values != frame['link_id'].iloc[2:].values)
        frame = frame[select]
    in_correspond = frame['link_id'].isin(correspond['link_id'])
    if np.count_nonzero(in_correspond) / len(frame) < road_exist_ratio:
        return pd.DataFrame(columns=frame.columns)
    frame = frame[in_correspond]
    if len(frame) < min_p or len(frame) > max_p:
        return pd.DataFrame(columns=frame.columns)
    return frame


def clean_line1(root_futian, root_nanshan, dates, correspond,
                output_root, futian_code='440304', nanshan_code='440305'):
    """
    初步清理轨迹1，包括筛选折线轨迹（同_clean_line1）、时间戳处理、
    去除大量经过小路（即匹配到的道路不在`correspond`中）的数据
    """
    def bigint_to_timestamp(bigint):
        """将大整数类型（格式yyyyMMddhhmmss，如20171001092515）转成时间戳"""
        return int(time.mktime((
            bigint // 10000000000, (bigint // 100000000) % 100, (bigint // 1000000) % 100,
            (bigint // 10000) % 100, (bigint // 100) % 100, bigint % 100,
            0, 0, 0)))

    correspond = pd.read_csv(correspond)
    dates = tqdm(dates)
    for day in dates:
        dates.set_description(day)
        trajs = []
        for _root, _code in [(root_futian, futian_code), (root_nanshan, nanshan_code)]:
            with gzip.open(Path(_root) / f'{_code}_{day}.gz') as gz:
                traj = pd.read_csv(gz, header=None)
            traj.columns = traj_header
            traj = traj[['trace_id', 'tm', 'speed', 'x', 'y', 'link_id']]
            trajs.append(traj)
        del gz, traj
        trajs = pd.concat(trajs, ignore_index=True)
        trajs['tm'] = trajs['tm'].apply(bigint_to_timestamp)
        trajs = pd.concat(jl.Parallel(8)(jl.delayed(_clean_line1)(frame, correspond)
                                         for _, frame in trajs.groupby('trace_id')))
        trajs.to_csv(Path(output_root) / f'{day}.csv', index=False)


def clean_poi1(poi_file, distance=0.5):
    """
    对POI类别重分类，并统计研究区内各类POI的数量。
    注：预处理时发现，有些POI地址只精确到大楼，所以楼内的所有POI共享一个地址，
    这些POI之间的距离是0。预处理过程会把距离 < `distance`的POI，根据POI类别的TF-IDF值，取最显著的一类

    深圳：auto_service: 2515, cultural: 3189, daily_life: 10273, enterprise: 7145,
    food: 15138, government: 1378, medical: 2919, recreation: 1744,
    residence: 6040, shopping: 19268, tourist: 747, transport: 59

    :param distance: 合并多个POI时，POI之间的距离阈值(m)（DBSCAN算法的半径）
    """
    shp = gpd.read_file(poi_file)
    shp['poiType'] = [''] * len(shp)
    for key in POI_dict:
        for regexp in POI_dict[key]:
            match = shp['POI_TYPE'].map(lambda s: re.match(regexp, s) is not None)
            shp.loc[match, 'poiType'] = key
    shp = shp[shp['poiType'] != '']
    shp.reset_index(drop=True, inplace=True)

    # 统计不同类型POI的重要性
    idf_ = shp.groupby('poiType').count().loc[POI_dict.keys(), 'POI']
    idf_ = np.log2(len(shp) / idf_)

    # 用DBSCAN找到距离很近的POI团簇，然后每个团簇仅取一个点
    shp4526 = shp.to_crs(PROJ_EPSG)
    points = np.stack([shp4526.geometry.x, shp4526.geometry.y], axis=1)
    dbscan = DBSCAN(eps=distance, min_samples=2, n_jobs=8).fit_predict(points)
    del_ind = np.zeros(dbscan.shape, np.bool_)
    for i in range(np.max(dbscan) + 1):
        _index = dbscan == i
        del_ind[_index] = True
        _poi = shp.iloc[_index]
        _poistat = _poi.groupby('poiType').count()
        _tf = pd.Series(data=[0] * len(idf_), index=idf_.index)
        _tf.loc[_poistat.index] = _poistat['POI']
        _maxType = (_tf * idf_).idxmax()
        del_ind[_poi[_poi['poiType'] == _maxType].index[0]] = False

    shp = shp[~del_ind]
    print(shp.groupby('poiType').count()['POI'])
    shp.to_file(Path(poi_file).with_name('poi_1.shp'))


def road_joinPOI(road_file, poi_file, radius=250, score='tfidf', output_name=None):
    """
    统计路网附近的各类型POI数量

    :param radius: 缓冲区半径(m)，在EPSG:4526投影下。
        默认的250m半径等价于500m直径，与500m格网一致（Zhipeng Gui et al., 2021）
    :param score: 统计的POI的方式。'num': 统计各类POI的绝对数量；
                    'tfidf': 统计各类POI的TF-IDF得分和POI总数
    """
    if output_name is None:
        output_name = f'road_poi{radius}_{score}.shp'
    road = gpd.read_file(road_file)
    road4526 = road.to_crs(PROJ_EPSG).buffer(radius)
    road4526 = gpd.GeoDataFrame(road[['Mesh_Code', 'ROAD_ID']], geometry=road4526, crs=PROJ_EPSG)
    poi = gpd.read_file(poi_file).to_crs(PROJ_EPSG)
    join = road4526.sjoin(poi[['poiType', 'geometry']])
    join = join.groupby(['Mesh_Code', 'ROAD_ID', 'poiType'], as_index=False).count()

    def _stat(name, group: pd.DataFrame):
        """统计每个路段内，各类别POI的数量"""
        _df = {'Mesh_Code': [name[0]], 'ROAD_ID': [name[1]]}
        for key in POI_dict:
            _data = group[group['poiType'] == key]
            _df[key] = [0] if _data.empty else _data['index_right'].values
        return pd.DataFrame(_df)

    join = pd.concat(jl.Parallel(8)(jl.delayed(_stat)(name, group)
                     for name, group in join.groupby(['Mesh_Code', 'ROAD_ID'])),
                     ignore_index=True)
    join['numPOI'] = join[POI_dict.keys()].sum(axis=1)

    # 根据POI数量，是统计什么指数？
    # 路段i类别j的TF-IDF = n_ij / n_i * log(N / N_j)
    if score == 'tfidf':
        tf_ = join[POI_dict.keys()].values / join[['numPOI']].values
        idf_ = poi.groupby('poiType').count().loc[POI_dict.keys(), 'POI']
        idf_ = np.log2(len(poi) / idf_.values[None, :])
        join.loc[:, POI_dict.keys()] = tf_ * idf_

    # 将统计好的POI指数，合并回原来的shp文件，然后填充空值（0）
    road_new = road.merge(join, on=['Mesh_Code', 'ROAD_ID'], how='left')
    road_new.fillna({k: 0 for k in list(POI_dict) + ['numPOI']}, inplace=True)
    if score in ('num',):
        road_new = road_new.astype({k: np.int64 for k in list(POI_dict)})
    road_new['numPOI'] = road_new['numPOI'].astype(np.int64)
    road_new.to_file(Path(road_file).with_name(output_name))


def word2vec_poi_pair(poi_file, max_distance=500, knn=10, weight='equal',
                      output_name=None, **kwargs):
    """
    根据POI几何点的距离，构建<center, context>POI对，用于word2vec训练POI类别向量。
    构建POI对时，同时考虑最大阈值距离、center的K最邻近POI。两个条件需同时满足（逻辑与）。
    默认500m最大范围 + KNN=10的参数来自于Liu Xi et al. (2019) [Place niche]

    :param max_distance: 最大距离阈值(m)
    :param knn: 对于同一个center，最多仅取前K个POI作为context
    :param weight: POI样本对的权重（等价于Place2vec中的样本重复次数beta）：
        'equal': 所有POI样本对具有相等的权重；
        'inv_distance': 按照距离倒数，设置权重（该方法可能不适合POI数据，
            因为POI在商贸区的密度很大，甚至最邻近POI不到10m，而城郊的最邻近POI可达上百米）；
        'flat_inv_distance': 当距离小于`flat_distance`(m)时，权重值是常数，大于`flat_distance`时，
            权重值按照距离倒数衰减。default: flat_distance = 50；
        'relative_inv_distance': 对于同一个center的不同context，按照距离倒数衰减，
            而同一center的权重总和不变（因此当POI对的距离相等时，可能仍有不同的权重）
        'reciprocal_rank': 权重值为排名倒数。对于同一个center的不同样本，按照距离排序
    """
    assert max_distance is not None or knn is not None
    assert weight in ('equal', 'inv_distance', 'flat_inv_distance',
                      'relative_inv_distance', 'reciprocal_rank')
    if output_name is None:
        typedict = {'equal': 'eq', 'inv_distance': 'inv', 'flat_inv_distance': 'flat',
                    'relative_inv_distance': 'rel', 'reciprocal_rank': 'rrank'}
        output_name = f'POIpair_{max_distance}_{knn}_{typedict[weight]}.csv'

    poi = gpd.read_file(poi_file).to_crs(PROJ_EPSG)
    points = np.stack([poi.geometry.x, poi.geometry.y], axis=1)
    poiType = np.array(poi['poiType'])
    poiid = np.array(poi['POI_ID'])
    batch = int(3e8 / len(points))
    result = []

    # 由于整个POI表太大，两两计算距离矩阵可能爆内存，因而分batch
    pbar = tqdm(range(0, len(points), batch))
    for b in pbar:
        pbar.set_description(f'batch {b // batch}')
        _data = points[b:b+batch, :]
        _dist = cdist(_data, points, 'euclidean')
        index = np.argsort(_dist, axis=1)

        # 仅筛选前k条<center, context>POI对（如果设置了knn），
        # 取1:是因为第0个一定是自身
        index = index[:, 1:knn+1] if knn is not None else index[:, 1:]
        _dist = np.take_along_axis(_dist, index, axis=1)
        id_y = np.broadcast_to(poiid[None, :], (_dist.shape[0], poiid.shape[0]))
        id_y = np.take_along_axis(id_y, index, axis=1)
        id_x = np.broadcast_to(poiid[b:b+batch, None], _dist.shape)
        label_y = np.broadcast_to(poiType[None, :], (_dist.shape[0], poiType.shape[0]))
        label_y = np.take_along_axis(label_y, index, axis=1)
        label_x = np.broadcast_to(poiType[b:b+batch, None], _dist.shape)
        label = np.stack([id_x, id_y, label_x, label_y], axis=2)
        del id_x, id_y, label_x, label_y, index

        # 按照距离阈值筛选POI对
        if max_distance is not None:
            index2 = _dist <= max_distance
            _dist = _dist[index2]
            label = label[index2, :]
            del index2
        _dist = _dist.reshape((-1,))
        label = label.reshape((-1, 4))
        result.append(pd.DataFrame({'center_id': label[:, 0], 'context_id': label[:, 1],
                                    'center': label[:, 2], 'context': label[:, 3],
                                    'weight': _dist}))

    result = pd.concat(result, axis=0, ignore_index=True)
    # 计算POI对的权重，_dist+1是防止除数过小导致权重过大
    if weight == 'equal':
        result['weight'] = 1
    elif weight == 'inv_distance':
        result['weight'] = max_distance / 10 / (result['weight'] + 1)
    elif weight == 'flat_inv_distance':
        kwargs.setdefault('flat_distance', 50)
        result['weight'] = np.where(result['weight'] < kwargs['flat_distance'], 1,
                                    kwargs['flat_distance'] / result['weight'])
    elif weight == 'relative_inv_distance':
        result['weight'] = 1 / (result['weight'] + 1)

        def _rel_inv_dist(g):
            g['weight'] = g['weight'] / g['weight'].mean()
            return g

        result = result.groupby('center_id').apply(_rel_inv_dist)
    elif weight == 'reciprocal_rank':
        def _recipr_rank(g):
            g['weight'] = 1 / np.arange(1, len(g) + 1, dtype=np.float_)
            return g

        result = result.groupby('center_id').apply(_recipr_rank)
    else:
        raise ValueError('Argument `weight` NOT available.')
    result.to_csv(Path(poi_file).with_name(output_name), index=False)


def generate_topo(shpfile, output_name=None, distance=1):
    """
    从路网生成路口（路网的交点node）拓扑关系。
    注：路网shp中的FNODE和TNODE不是唯一标识。要结合Mesh_Code后，<Mesh_Code, NODE_ID>才是唯一的标识。
    但这样的问题是，一个地理点可能刚好在Mesh的边界处，从而多个<Mesh_Code, NODE_ID>键对应同一个点。

    实际上的算法是：每个路段都生成1个首点和1个尾点，然后对这些点聚类，找到空间上的同一个点，
    然后建立路段的拓扑邻接关系。
    """
    if output_name is None:
        output_name = Path(shpfile).stem + '_roadlink.csv'

    shp = gpd.read_file(shpfile)
    nodes = []
    for line in shp.itertuples():
        x, y = line.geometry.xy
        nodes.append({'road1_id': line.road1_id, 'is_end': False,
                      'geometry': geo.Point(x[0], y[0])})
        nodes.append({'road1_id': line.road1_id, 'is_end': True,
                      'geometry': geo.Point(x[-1], y[-1])})
    nodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs=shp.crs)
    nodes1 = nodes.to_crs(PROJ_EPSG)
    points = np.stack([nodes1.geometry.x, nodes1.geometry.y], axis=1)
    nodes['cluster'] = DBSCAN(eps=distance, min_samples=1, n_jobs=8).fit_predict(points)
    print('node cluster: finish.')

    # 根据聚类结果，聚合相同地理位置的边
    connect_list = []
    for _, group in nodes.groupby('cluster'):
        for line in group.itertuples():
            others = group.loc[group['road1_id'] != line.road1_id, 'road1_id'].values
            connect_list.append({'road1_id': line.road1_id,
                                 'connect': ' '.join(others.astype(str)),
                                 'node_lon': line.geometry.xy[0][0],
                                 'node_lat': line.geometry.xy[1][0],
                                 'is_end': line.is_end})
    connect_list = pd.DataFrame(connect_list)
    print('topology of road segment: finish.')

    connect_list = connect_list.groupby('road1_id', as_index=False).apply(
        lambda g: pd.Series({'Flon': g.loc[~g['is_end'], 'node_lon'].iloc[0],
                             'Flat': g.loc[~g['is_end'], 'node_lat'].iloc[0],
                             'Tlon': g.loc[g['is_end'], 'node_lon'].iloc[0],
                             'Tlat': g.loc[g['is_end'], 'node_lat'].iloc[0],
                             'Fconnect': g.loc[~g['is_end'], 'connect'].iloc[0],
                             'Tconnect': g.loc[g['is_end'], 'connect'].iloc[0]}))
    connect_list.to_csv(Path(shpfile).with_name(output_name), index=False)


def clean_road2(shpfile, correspond_path, output_name):
    """
    把路网数据规范至embedding层可以接收的输入（one hot），整理需要输入的特征。
    注：'numPOI'列是区域附近的POI数量，该值做了[min, max]映射到[0, 1]
    """
    roads = gpd.read_file(shpfile)
    columns = list(roads.columns)
    columns[12] = 'auto_service'
    roads.columns = columns
    correspond = pd.read_csv(correspond_path)
    roads = roads.merge(correspond, left_on=['Mesh_Code', 'ROAD_ID'], right_on=['mesh_code', 'road_id'])
    roads['road_class'] = roads['ROAD_CLASS'].map(ROADCLASS_map)
    roads['direction'] = roads['DIRECTION'] - 1
    roads['form_way'] = roads['FORM_WAY'].map(FORMWAY_map)
    roads['link_type'] = roads['LINK_TYPE'].map(LINKTYPE_map)
    roads['numPOI'] = roads['numPOI'] / roads['numPOI'].max()
    roads = roads[['link_id', 'road1_id', 'road_class', 'direction', 'form_way',
                   'link_type', 'numPOI', *POI_dict.keys()]].copy()
    roads.to_csv(Path(shpfile).with_name(output_name), index=False)


def clean_line2(root_traj, road_file, output_dir):
    """
    轨迹标签重分配：使用重新分配的道路ID，文件内轨迹ID，投影经纬度，k-fold。
    最后生成一个所有轨迹的基本元数据表。
    注：速度特征保留了，
    """
    def _stat_time(t):
        st = time.localtime(t)
        date = dt.date(st.tm_year, st.tm_mon, st.tm_mday)
        return (st.tm_mon, st.tm_mday, st.tm_wday, int(date in VACATION),
                st.tm_hour, st.tm_min, st.tm_sec)

    roads = pd.read_csv(road_file, usecols=['link_id', 'road1_id'])
    meta = []
    record_cnt = 0
    files = Path(root_traj).iterdir()
    files = tqdm(list(filter(lambda v: v.is_file and re.match('.csv', v) is not None, files)))
    for file in files:
        files.set_description(f'process {file.name}')
        traj = pd.read_csv(file)
        # 重分配ID、经纬度坐标变换、链接路网、速度值放缩（认为最高速度120，缩放至[0, 1]）
        unique_id = traj['trace_id'].unique()
        traj['trace_id'] = traj['trace_id'].map({
            k: v for k, v in zip(unique_id, range(record_cnt, record_cnt + len(unique_id)))})
        record_cnt += len(unique_id)
        traj = traj.merge(roads, on='link_id', how='left')
        assert np.all(~traj['road1_id'].isna())
        del traj['link_id']
        traj['speed'] /= 120
        traj[['x', 'y']] = np.concatenate(jl.Parallel(8)(
            jl.delayed(geopoint_to_mp1)(traj[['x', 'y']].iloc[i: i+len(traj)//8].values)
            for i in range(0, len(traj), len(traj) // 8)), axis=0)

        # 统计每个轨迹的元数据信息
        groupby = traj.groupby('trace_id', as_index=False)
        _meta = groupby.apply(lambda g: pd.Series(
            {'start_time': g['tm'].min(), 'end_time': g['tm'].max(),
             'num_record': len(g), 'dest_road': g['road1_id'].iloc[-1]}))
        _meta1 = groupby.apply(lambda g: pd.Series(
            {'numPOI': g['numPOI'].iloc[0], **{k: g[k].iloc[0] for k in POI_dict.keys()},
             'trip_len': np.sqrt(np.square(np.diff(g[['x', 'y']], axis=0)).sum(axis=1)).sum(),
             'dest_x': g['x'].iloc[-1], 'dest_y': g['y'].iloc[-1]}))
        del _meta1['trace_id']
        _meta['k_fold'] = shuffle(np.arange(len(_meta)) % 5)
        _meta2 = pd.DataFrame(list(_meta['start_time'].map(_stat_time)),
                              columns=['month', 'day', 'weekday', 'is_vacation',
                                       'start_h', 'start_m', 'start_s'])
        _meta = pd.concat([_meta, _meta1, _meta2], axis=1)
        meta.append(_meta)
        traj.to_csv(Path(output_dir) / file.name, index=False)

    meta = pd.concat(meta, ignore_index=True)
    meta.to_csv(Path(output_dir) / 'metadata.csv', index=False)


def clean_line3(root_traj, road_file):
    """后来发现，在元数据表中，需要把出发地的POI信息单独列出，因而使用该新函数"""
    root_traj = Path(root_traj)
    meta = pd.read_csv(root_traj / 'metadata.csv')
    roads = pd.read_csv(road_file, index_col='road1_id')
    files = list(filter(lambda v: v.is_file and re.match(r'[0-9]{8}\.csv', v.name) is not None,
                        root_traj.iterdir()))
    # files = tqdm(files)
    _new_meta = pd.concat(
        jl.Parallel(4, verbose=11)(jl.delayed(_clean_line3)(file, roads)
                                   for file in files), ignore_index=True)
    meta = meta.merge(_new_meta, on='trace_id')
    meta.to_csv(root_traj / 'metadata_new.csv', index=False)


def _clean_line3(file, roads):
    traj = pd.read_csv(file)
    groupby = traj.groupby('trace_id', as_index=False)
    _meta1 = groupby.apply(lambda g: pd.Series(
        # {k: roads.loc[g['road1_id'].iloc[0], k]
        #  for k in ['numPOI'] + list(POI_dict.keys())}))
        {'trip_len': np.sqrt(np.square(np.diff(g[['x', 'y']], axis=0)).sum(axis=1)).sum()}))
    return _meta1


def cluster_destination(meta_file, kfold=(0, 1, 2), bandwidth=0.01, output_name=None):
    """
    从指定的轨迹目的地中，使用Mean-Shift聚类出一系列聚集区域
    """
    meta_file = Path(meta_file)
    output_name = output_name if output_name is not None else 'dest_cluster'
    df = pd.read_csv(meta_file, usecols=['k_fold', 'dest_x', 'dest_y'])
    if kfold is None:
        kfold = tuple(range(5))
    df = df[df['k_fold'].isin(kfold)]
    model = MeanShift(bandwidth=bandwidth)
    model.fit(df[['dest_x', 'dest_y']].to_numpy())
    np.save(meta_file.with_name(f'{output_name}.npy'), model.cluster_centers_)
    return model.cluster_centers_


if __name__ == '__main__':
    # 合并两个城区的路网和POI
    # merge_district_roadPOI()
    # 去除路网中的小路
    # clean_road1(TRAJ_ROOT / 'roads/road_network.shp',
    #             TRAJ_ROOT / 'roads/correspondence.csv')
    # 初步清理不满足要求的轨迹
    # clean_line1(TRAJ_ROOT / '福田区GPS轨迹', TRAJ_ROOT / '南山区GPS轨迹',
    #             [f'201710{day:02d}' for day in list(range(1, 25)) + list(range(26, 32))]
    #             + [f'201711{day:02d}' for day in range(1, 31)]
    #             + [f'201712{day:02d}' for day in range(1, 32)],
    #             TRAJ_ROOT / 'roads/correspondence_1.csv',
    #             TRAJ_ROOT / 'trajectories/backup/')

    # 清理POI数据、准备训练place2vec的POI对
    # clean_poi1(TRAJ_ROOT / 'roads/poi.shp')
    # road_joinPOI(TRAJ_ROOT / 'roads/road_1.shp',
    #              TRAJ_ROOT / 'roads/poi_1.shp', radius=250)
    # word2vec_poi_pair(TRAJ_ROOT / 'roads/poi_1.shp',
    #                   max_distance=500, knn=10, weight='flat_inv_distance')

    # generate_topo(TRAJ_ROOT / 'roads/road_1.shp',
    #               output_name='road_1_roadlink.csv')

    # 把路网数据规范至embedding层可以接收的输入（one hot），整理需要输入的特征
    # clean_road2(TRAJ_ROOT / 'roads/road_poi250_tfidf.shp',
    #             TRAJ_ROOT / 'roads/correspondence_1.csv',
    #             'road_input.csv')
    # clean_line2(TRAJ_ROOT / 'trajectories/backup/',
    #             TRAJ_ROOT / 'roads/road_input.csv',
    #             TRAJ_ROOT / 'trajectories/inputs/')
    # clean_line3(TRAJ_ROOT / 'trajectories/inputs/',
    #             TRAJ_ROOT / 'roads/road_input.csv')
    cluster_destination(TRAJ_ROOT / 'trajectories/inputs/metadata/metadata_october.csv')
