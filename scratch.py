# 测试代码，用于数据观察，制图……

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sb
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr
import shapely.geometry as geo
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.special import fdtr
import joblib as jl
from pathlib import Path
import time
from tqdm import tqdm
import torch
from globalval import *
from preprocess_sz import mp1_to_geopoint
from predict_trainval import _xy_distance, _resolution


def stat_road_heatmap(traj, correspond, road_seg):
    """
    统计路网中每个路段的热力分布，即每个路段有多少辆车经过

    :param traj: 轨迹文件
    :param correspond: 路网文件1（记录编号和矢量图形的映射）
    :param road_seg: 路网文件2（路段矢量数据）
    """
    roads_cnt = road_seg[['Mesh_Code', 'ROAD_ID']]
    roads_cnt = roads_cnt.merge(correspond, left_on=['Mesh_Code', 'ROAD_ID'], right_on=['mesh_code', 'road_id'])
    # print(roads_cnt)

    # 去除轨迹中的无效定位
    traj = traj[traj['speed'] < 180]
    # 车速较慢时，可能有多个定位点在同一路段上，需要去重（即一条轨迹内多个同样的轨迹点需要去重）
    traj_cnt = traj[['trace_id', 'link_id']].groupby('link_id').nunique()
    traj_cnt.columns = ['cnt']
    # print(traj_cnt, traj[['trace_id', 'link_id']].groupby('link_id').count(), sep='\n')
    print('count finish')
    traj_cnt = roads_cnt[['Mesh_Code', 'ROAD_ID', 'link_id']].merge(traj_cnt, left_on='link_id', right_index=True)
    # print(traj_cnt)
    traj_cnt = traj_cnt[['Mesh_Code', 'ROAD_ID', 'cnt']].groupby(['Mesh_Code', 'ROAD_ID']).sum()
    # print(traj_cnt)
    result = road_seg.merge(traj_cnt, left_on=['Mesh_Code', 'ROAD_ID'], right_index=True, how='left')
    result['cnt'] = result['cnt'].fillna(0).astype(int)
    # print(road_seg)
    print('finish')
    return gpd.GeoDataFrame(result, crs=road_seg.crs)


def generate_road_heatmap():
    """
    调用`stat_road_heatmap`，读取文件并将统计结果保存成shp
    """
    for code, region, pinyin in [('04', '福田', 'ft'), ('05', '南山', 'ns')]:
        traj = pd.read_csv(TRAJ_ROOT / f'{region}区GPS轨迹/4403{code}_20171001', header=None)
        traj.columns = traj_header
        roads1 = pd.read_csv(ROAD_SEG_ROOT / f'{region}区/correspondence_{pinyin}_17q4.txt', header=None)
        roads1.columns = correspondence_header
        roads2 = gpd.read_file(ROAD_SEG_ROOT / f'{region}区/RoadSegment_17q4.TAB')
        result = stat_road_heatmap(traj, roads1, roads2)
        result.to_file(TRAJ_ROOT / f'analyze/{pinyin}_1001.shp')


def cal_timeseries(trajs, title=None, legend=None):
    """
    计算轨迹的时谱曲线，不同时段的轨迹量

    :param traj: 多个轨迹文件组成的列表
    """
    plt.figure(figsize=(8, 4.8), dpi=200)
    for traj in trajs:
        tm = traj[['trace_id', 'tm']].copy()
        tm['tm'] = tm['tm'] % 1000000
        tm = tm.groupby('trace_id').median()
        tm = (tm / 10000).astype(int)
        cnt = np.zeros((24,), np.int_)
        for k, frame in tm.groupby('tm'):
            cnt[k] = frame.shape[0]
        plt.plot(np.arange(24), cnt)
    plt.gca().xaxis.set_major_locator(MultipleLocator(4))
    plt.xlabel('hour')
    plt.ylabel('number of traces')
    plt.title(title)
    if legend is not None:
        plt.legend(legend)
    plt.show()


def generate_timeseries_plot():
    """
    调用`cal_timeseries`生成时间序列曲线图
    """
    files = []
    for code, region in [('04', '福田'), ('05', '南山')]:
        for time in (11, 10):
            df = pd.read_csv(TRAJ_ROOT / f'{region}区GPS轨迹/4403{code}_2017{time}01', header=None)
            df.columns = traj_header
            files.append(df)
    del df
    cal_timeseries(files, title=f'time series of two districts',
                   legend=['Futian weekday', 'Futian holiday', 'Nanshan weekday', 'Nanshan holiday'])


def generate_p(frame, latname, lonname):
    '''生成一个点要素'''
    return geo.Point(frame[lonname], frame[latname])


def generate_line(frame, latname, lonname):
    '''
    生成一条折线轨迹，轨迹以一个frame表示，每行表示一个点。未达到要求返回None

    :param min_p: 最少的记录点数
    :param max_interval: 两个轨迹点之间的最长时间间隔(s)
    '''
    # 目前的问题：1. 有些轨迹记录其实明显不是单条，而是在空间上多段的。
    # 但是也有过隧道导致信号中断的，如何区分？注：南山区东滨隧道用时4min左右，
    x, y = list(frame[lonname]), list(frame[latname])
    return geo.LineString(zip(x, y))


def cal_distance_hist(traj: pd.DataFrame, title=None):
    '''统计轨迹的出行距离、记录点数'''
    geos = traj.groupby('trace_id').apply(generate_line, 'y', 'x')
    geos.dropna(inplace=True)
    geos = gpd.GeoSeries(geos, crs='epsg:4326')
    geos = geos.to_crs(PROJ_EPSG)
    plt.figure(dpi=200)
    hist = plt.hist(geos.length / 1000, bins=100, range=[0, 30])
    print(hist)
    print('point cnt:', 'max:', np.max(geos.apply(lambda x: len(x.xy[0]))))
    plt.xlabel('distance (km)')
    plt.ylabel('number of traces')
    plt.title(title)
    plt.show()
    return gpd.GeoDataFrame({'geometry': geos, 'length': geos.length})


def cal_distance_stat():
    '''统计轨迹的出行距离、记录点数，包括文件读取、合并、筛选部分'''
    trajs = []
    for code, region, pinyin in [('04', '福田', 'Futian'), ('05', '南山', 'Nanshan')]:
        traj = pd.read_csv(TRAJ_ROOT / f'{region}区GPS轨迹/4403{code}_20171101', header=None)
        traj.columns = traj_header
        trajs.append(traj)
    del traj
    trajs = pd.concat(trajs)
    minmax = trajs.groupby('trace_id').tm.aggregate(['min', 'max'])
    print('max_time:', np.max(minmax['max'] - minmax['min']))
    del minmax
    geos = cal_distance_hist(trajs, title=f'histogram of travel distance in merged data')
    geos['trace_id'] = geos.index
    geos.to_file(TRAJ_ROOT / f'analyze/merged_trace_1101_cleanwalk.shp', index=False)


def vector_look(points, title_scatter='', metric='euclidean', items=None,
                corr_mat=False, title_corr='', pca_eigen=False, title_eigen=None):
    '''
    查看向量的统计信息、散点图，（可选）绘制相关性矩阵，（可选）绘制向量的主成分特征值累计信息量
    '''
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    print('norm', {'max': np.max(norm), 'min': np.min(norm), 'mean': np.mean(norm)})
    normdata = points / norm
    # 归一化方式：是否需要将散点整体平移至(0, 0)
    # normdata = (normdata - np.mean(normdata, axis=0, keepdims=True)) / np.linalg.norm(normdata, axis=0, keepdims=True)
    # normdata = normdata / np.linalg.norm(normdata, axis=1, keepdims=True)

    # 散点图
    if metric.lower() == 'pca':
        model = PCA(2)
    else:
        model = TSNE(metric=metric)
    data2d = model.fit_transform(points)
    plt.figure(dpi=150)
    plt.scatter(data2d[:, 0], data2d[:, 1])
    plt.title(title_scatter)
    plt.xlabel('x')
    plt.ylabel('y')
    # 注记
    if items is not None:
        for i in range(items.shape[0]):
            plt.annotate(items[i], xy=data2d[i, :], xytext=data2d[i, :] + 0.02)

    # 相关性矩阵
    if corr_mat:
        sim = np.dot(normdata, normdata.T)
        print('correlation matrix', {'max': np.max(sim), 'min': np.min(sim), 'mean': np.mean(sim)})
        plt.figure(figsize=(8, 6), dpi=150)
        annotation = np.array([f'{v:.3f}' for v in sim.flatten()]).reshape(sim.shape)
        ax = sb.heatmap(sim, cmap=plt.get_cmap('RdBu_r'), center=0, annot=annotation, fmt='s',
                        xticklabels=items, yticklabels=items, cbar_kws={'fraction': 0.05})
        ax.set_title(title_corr)
        plt.tight_layout()

    # 主成分信息量累计
    if pca_eigen:
        pca = PCA().fit(points)
        cumsv = np.cumsum(pca.singular_values_ ** 2)
        cumsv = cumsv / cumsv[-1]
        plt.figure(dpi=150)
        plt.plot(np.arange(pca.n_components_) + 1, cumsv)
        plt.title(title_eigen)
        plt.xlabel('principle component')
        plt.ylabel('cumulated explain ratio')
    plt.show()


def embedding_3dcolor(embedding, metric='euclidean', output=None):
    """使用TSNE降维方式，将嵌入向量降维至3D空间，并使用RGB颜色表示这3维"""
    model = TSNE(3, metric=metric, n_jobs=8)
    data3d = model.fit_transform(embedding)
    data3d = 255 - (np.max(data3d, axis=0, keepdims=True) - data3d) * 255 / \
             np.max(np.max(data3d, axis=0) - np.min(data3d, axis=0))
    data3d = data3d.astype(np.int_)
    df = pd.DataFrame({'road1_id': np.arange(len(embedding)), 'color_r': data3d[:, 0],
                      'color_g': data3d[:, 1], 'color_b': data3d[:, 2]})
    df['color'] = '#' + df["color_r"].apply(lambda l: f'{l:02x}') \
                      + df["color_g"].apply(lambda l: f'{l:02x}') \
                      + df["color_b"].apply(lambda l: f'{l:02x}')
    df.to_csv(output, index=False)
    return data3d


def draw_loss_curve(file, title=None, xlabel=None, ylabel=None, twinx=None, column_dict=None):
    """
    绘制训练过程中loss、一些指标值的变化曲线。

    :param twinx: 画在副y轴的变量
    :param column_dict: 从column name到legend label的映射
    """
    df = pd.read_csv(file)
    print(df.columns)
    fig = plt.figure(dpi=150)
    ax = plt.gca()
    if twinx is not None:
        ax1 = ax.twinx()
    lines = []
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(1, len(df.columns)):
        if twinx is None or df.columns[i] not in twinx:
            line_, = ax.plot(df['epoch'], df.iloc[:, i], colors[(i - 1) % len(colors)])
        else:
            line_, = ax1.plot(df['epoch'], df.iloc[:, i], colors[(i - 1) % len(colors)])
        if column_dict is not None and df.columns[i] in column_dict:
            lines.append((line_, column_dict[df.columns[i]]))
        else:
            lines.append((line_, df.columns[i]))
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    if twinx is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel[0])
        ax1.set_ylabel(ylabel[1])
    plt.legend(*list(zip(*lines)))
    plt.tight_layout()
    plt.show()


def stat_label_distribution(traj_meta, output_geostat=None, title=None,
                            xlabel=None, ylabel=None, subplot_xmax=None):
    """
    统计所有轨迹标签的分布：频数分布和空间分布。
    轨迹的目的地很可能是不均匀的，部分区域、部分路段被许多轨迹访问。
    """
    meta = pd.read_csv(traj_meta)
    bins = np.bincount(meta['dest_road'])
    cumbins = np.cumsum(-np.sort(-bins))
    cumbins = cumbins / cumbins[-1]
    plt.figure(dpi=150)
    plt.grid(linestyle=':')
    plt.plot(np.arange(len(cumbins)), cumbins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if subplot_xmax is not None:
        plt.axes([0.3, 0.2, 0.5, 0.4])
        plt.plot(np.arange(subplot_xmax), cumbins[:subplot_xmax])
        plt.grid(linestyle=':')
    plt.show()
    if output_geostat is not None:
        df = pd.DataFrame({'road1_id': np.arange(len(bins)), 'label_count': bins})
        df.to_csv(output_geostat, index=False)


def draw_val_acc(df):
    """绘制不同模型的训练集损失、验证集指标"""
    new_df = [{'model': line.model, 'index': col, 'value': line._asdict()[col]}
              for line in df.itertuples() for col in df.columns[1:]]
    df = pd.DataFrame(new_df)
    df1 = df.copy()
    df1.loc[df1['index'] != 'train_loss_entr', 'value'] = 0
    df2 = df.copy()
    df2.loc[df2['index'] == 'train_loss_entr', 'value'] = 0
    plt.figure(dpi=150)
    ax = plt.gca()
    ax1 = ax.twinx()
    sb.barplot(data=df1, x='index', y='value', hue='model', ax=ax)
    sb.barplot(data=df2, x='index', y='value', hue='model', ax=ax1)
    ax.get_legend().remove()
    plt.title('train & val indices on prediction models')
    ax.set_xlabel('train & val accuracy indices')
    ax.set_ylabel('train loss')
    ax1.set_ylabel('val accuracy index')
    plt.legend(loc=4)
    plt.show()


def draw_err_hist(files: list, keys: list, bins=100, range=None, title=None, xlabel=None, legend=None):
    """绘制误差统计直方图"""
    arrs = []
    legend = [None] * len(files) if legend is None else legend
    for f, k, lgd in zip(files, keys, legend):
        data = np.load(f)[k]
        lgd = lgd if lgd is not None else k
        print(f'{lgd}: mean={np.mean(data)}, median={np.median(data)}, '
              f'acc_within650={np.count_nonzero(data < 650) / data.shape[0]}')
        arrs.append(pd.DataFrame({'x': data, 'h': [lgd] * data.shape[0]}))
    arrs = pd.concat(arrs, ignore_index=True)
    plt.figure(dpi=150)
    sb.histplot(x=arrs['x'].to_numpy(), hue=arrs['h'].to_numpy(),
                stat='proportion', bins=bins, binrange=range, element='poly')
    # plt.hist(arrs, bins=bins, range=range, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    # 子图
    plt.axes([0.3, 0.25, 0.55, 0.5])
    sb.histplot(x=arrs['x'].to_numpy(), hue=arrs['h'].to_numpy(),
                stat='proportion', bins=bins, binrange=range, element='poly', legend=False)
    plt.xlim(left=range[0] + 500)
    plt.ylim(top=0.025)
    plt.ylabel('')
    plt.grid(linestyle=':')
    plt.show()


def sample_pred_distribution(data_root, road_data, model_path, output_shp,
                             test_fn='call', trace_id=None, proportions=None, repeat=32):
    """在地图上绘制示例数据下，模型的预测结果"""
    assert test_fn in ['call', 'p_sample']

    from predict_dataset import TrajectoryDataSingleSample
    from predict_trainval import DEVICE, _to_device

    if isinstance(proportions, (int, float)):
        proportions = [proportions]
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    dataset = TrajectoryDataSingleSample(data_root, road_data, len_ratio=proportions[0])
    if trace_id is None:
        trace_id = np.random.choice(dataset.meta.index)
    points_df = {'model': [], 'param': [], 'geometry': []}
    lines_df = {'trace_id': [], 'param': [], 'geometry': []}
    for prop in proportions:
        dataset.len_ratio = prop
        x, y = _to_device(dataset[trace_id], DEVICE)
        with torch.inference_mode():
            if test_fn == 'call':
                pred = model(*x, predict_final=True, pred_y_index=slice(0, 3))[0]
                pred = torch.cat(pred, dim=0)
            else:
                pred = model.p_sample_loop_repeat(
                    *x, predict_final=True, pred_y_index=1, repeat=repeat).squeeze(0)
            pred = mp1_to_geopoint(pred.cpu().numpy())
        point_pred = [geo.Point(pred[i, :]) for i in range(pred.shape[0])]
        points_df['model'].extend([Path(model_path).stem] * len(point_pred))
        points_df['param'].extend([prop] * len(point_pred))
        points_df['geometry'].extend(point_pred)
        line = geo.LineString(mp1_to_geopoint(x[3].cpu().numpy()[:, 0, :2]))
        lines_df['trace_id'].append(trace_id)
        lines_df['param'].append(prop)
        lines_df['geometry'].append(line)
    point_true = geo.Point(mp1_to_geopoint(dataset[trace_id][1][0])[0, :])
    points_df['model'].append('ground truth')
    points_df['param'].append(1)
    points_df['geometry'].append(point_true)
    output_shp = Path(output_shp)
    df_p = gpd.GeoDataFrame(points_df, crs=PROJ_EPSG)
    df_p.to_file(output_shp.with_name(output_shp.stem + '_points.shp'))
    df_l = gpd.GeoDataFrame(lines_df, crs=PROJ_EPSG)
    df_l.to_file(output_shp.with_name(output_shp.stem + '_line.shp'))


def analyze_pred_err_distr(npz_files, metadata):
    """分析预测误差在不同轨迹长度/已知轨迹比例/时空的分布"""
    dfs = []
    for k, npz_file in npz_files.items():
        npz = np.load(npz_file)
        err = _xy_distance(torch.as_tensor(npz['y_pred']),
                           torch.as_tensor(npz['y_true'])).numpy()
        df = pd.DataFrame({'err': err / _resolution / 1000, 'err_rel': err / npz['others'][:, 0],
                           'trace_id': npz['others'][:, 1].astype(np.int_),
                           'prefix_len': npz['others'][:, 2].astype(np.int_)})
        df['Model'] = k
        dfs.append(df)
        # if is_psample:
        #     df['scatter'] = np.sqrt(np.var(npz['p_sample'], axis=1).sum(axis=-1)) / _resolution / 1000
    df = pd.concat(dfs, ignore_index=True)
    meta = pd.read_csv(metadata)
    df = df.merge(meta, on='trace_id')
    df['prefix_ratio'] = df['prefix_len'] / df['num_record']
    df['trip_len'] = df['trip_len'] / _resolution / 1000
    # agg = df.groupby(['model', 'cut'], as_index=False).agg(
    #     mean_dist=pd.NamedAgg('err', np.mean),
    #     std_dist=pd.NamedAgg('err', np.std),
    #     median_dist=pd.NamedAgg('err', np.median),
    #     acc_500m=pd.NamedAgg('err', lambda g: np.count_nonzero(g < 0.5) / g.shape[0]),
    #     acc_1km=pd.NamedAgg('err', lambda g: np.count_nonzero(g < 1.0) / g.shape[0]),
    #     mean_rel_err=pd.NamedAgg('err_rel', np.mean),
    #     median_rel_err=pd.NamedAgg('err_rel', np.median),
    # )
    # 尝试看看误差的分布
    plt.figure(dpi=600)
    sb.histplot(df, x='err', hue='Model', binrange=(0, 15),
                stat='proportion', element='poly')
    plt.xlabel('Prediction error (km)')
    plt.xlim((0, 5))
    plt.tight_layout()
    ax = plt.axes([0.35, 0.25, 0.6, 0.55])
    sb.histplot(df, x='err', hue='Model', binrange=(0, 15),
                stat='proportion', element='poly', ax=ax, legend=False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xlim((1, 15))
    plt.ylim((0, 0.01))
    ax = plt.axes([0.55, 0.375, 0.4, 0.4])
    sb.histplot(df, x='err', hue='Model', binrange=(0, 15),
                stat='proportion', element='poly', ax=ax, legend=False)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xlim((5, 15))
    plt.ylim((0, 1e-3))
    # plt.savefig(TRAJ_ROOT / 'figures/figure5.svg')

    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2), dpi=300)
    # 轨迹前缀长度对预测精度的影响
    cut_range = np.linspace(0, 240, 13)
    df['cut'] = pd.cut(df['prefix_len'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2).astype(int))
    print(df.loc[df['Model'] == 'Diffusion model', 'cut'].value_counts())
    sb.pointplot(data=df, x='cut', y='err', hue='Model', estimator='median',
                 errorbar=None, dodge=False, markers=['o', 's'], ax=axs[0, 0])
    axs[0, 0].set_title('(a)')
    axs[0, 0].set_xlabel('length of trajectory prefix (record points)')
    axs[0, 0].set_ylabel('median error (km)')
    axs[0, 0].set_xticks(np.arange(cut_range.shape[0]) - 0.5, cut_range.astype(int))
    axs[0, 0].grid(linestyle=':')
    # 轨迹前缀比例对预测精度的影响
    cut_range = np.linspace(0.0, 1.0, 11)
    df['cut'] = pd.cut(df['prefix_ratio'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2, 2))
    print(df.loc[df['Model'] == 'Diffusion model', 'cut'].value_counts())
    sb.pointplot(data=df, x='cut', y='err', hue='Model', estimator='median',
                 errorbar=None, dodge=False, markers=['o', 's'], ax=axs[0, 1])
    axs[0, 1].set_title('(b)')
    axs[0, 1].set_xlabel('proportion of trajectory prefix')
    axs[0, 1].set_ylabel('median error (km)')
    axs[0, 1].set_xticks(np.arange(cut_range.shape[0]) - 0.5, np.round(cut_range, 2))
    axs[0, 1].grid(linestyle=':')
    # 轨迹长度对预测精度的影响
    cut_range = np.linspace(0, 22, 12)
    df['cut'] = pd.cut(df['trip_len'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2).astype(int))
    print(df.loc[df['Model'] == 'Diffusion model', 'cut'].value_counts())
    sb.pointplot(data=df, x='cut', y='err', hue='Model', estimator='median',
                 errorbar=None, dodge=False, markers=['o', 's'], ax=axs[1, 0])
    axs[1, 0].set_title('(c)')
    axs[1, 0].set_xlabel('distance from origin to destination (km)')
    axs[1, 0].set_ylabel('median error (km)')
    axs[1, 0].set_xticks(np.arange(cut_range.shape[0]) - 0.5, cut_range.astype(int))
    axs[1, 0].grid(linestyle=':')
    # 出发时间对预测精度的影响
    sb.pointplot(data=df, x='start_h', y='err', hue='Model', estimator='median',
                 errorbar=None, dodge=False, markers=['o', 's'], ax=axs[1, 1])
    print(df.loc[df['Model'] == 'Diffusion model', 'start_h'].value_counts())
    axs[1, 1].set_title('(d)')
    axs[1, 1].set_xlabel('departure time in a day (hour)')
    axs[1, 1].set_ylabel('median error (km)')
    axs[1, 1].grid(linestyle=':')
    axs[1, 1].set_xticks(np.linspace(0, 24, 7) - 0.5, np.linspace(0, 24, 7, dtype=int))
    # 日期对预测精度的影响
    # sb.pointplot(data=df, x='weekday', y='err', hue='model', estimator='median',
    #              errorbar=None, dodge=False, ax=axs[1, 1])
    # axs[1, 1].set_ylabel('median of prediction error (km)')
    # axs[1, 1].grid(linestyle=':')
    # axs[1, 1].set_xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    # plt.savefig(TRAJ_ROOT / 'figures/figure6.png')
    plt.show()


def analyze_psample_scatter(npz_file, metadata):
    """分析扩散模型预测结果时，目的地散布程度（标准距离）的影响因素"""
    npz = np.load(npz_file)
    err = _xy_distance(torch.as_tensor(npz['y_pred']),
                       torch.as_tensor(npz['y_true'])).numpy()
    scatter = np.sqrt(np.var(npz['p_sample'], axis=1).sum(axis=-1)) / _resolution / 1000
    df = pd.DataFrame({'err': err / _resolution / 1000, 'err_rel': err / npz['others'][:, 0],
                       'trace_id': npz['others'][:, 1].astype(np.int_),
                       'prefix_len': npz['others'][:, 2].astype(np.int_),
                       'scatter': scatter})
    meta = pd.read_csv(metadata)
    df = df.merge(meta, on='trace_id')
    df['prefix_ratio'] = df['prefix_len'] / df['num_record']
    df['trip_len'] = df['trip_len'] / _resolution / 1000

    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2), dpi=300)
    # 轨迹前缀长度对标准距离的影响
    cut_range = np.linspace(0, 240, 13)
    df['cut'] = pd.cut(df['prefix_len'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2).astype(int))
    sb.pointplot(data=df, x='cut', y='scatter', estimator='median',
                 errorbar=('pi', 50), ax=axs[0, 0])
    axs[0, 0].set_title('(a)')
    axs[0, 0].set_xlabel('length of trajectory prefix (record points)')
    axs[0, 0].set_ylabel('standard distance (km)')
    axs[0, 0].set_xticks(np.arange(cut_range.shape[0]) - 0.5, cut_range.astype(int))
    axs[0, 0].grid(linestyle=':')
    corr = df['prefix_len'].corr(df['scatter'])
    Ftest = np.square(corr) / (1 - np.square(corr)) * (df.shape[0] - 2)
    pval = max(1 - fdtr(1, df.shape[0] - 2, Ftest), 0.001)
    axs[0, 0].annotate(f'$r={corr:.3f}$\n$p{"<" if pval == 0.001 else "="}{pval:.3f}$',
                       (0.82, 0.9), xycoords='axes fraction')
    # 轨迹前缀比例对分布模式的影响
    cut_range = np.linspace(0.0, 1.0, 11)
    df['cut'] = pd.cut(df['prefix_ratio'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2, 2))
    sb.pointplot(data=df, x='cut', y='scatter', estimator='median',
                 errorbar=('pi', 50), ax=axs[0, 1])
    axs[0, 1].set_title('(b)')
    axs[0, 1].set_xlabel('proportion of trajectory prefix')
    axs[0, 1].set_ylabel('standard distance (km)')
    axs[0, 1].set_xticks(np.arange(cut_range.shape[0]) - 0.5, np.round(cut_range, 2))
    axs[0, 1].grid(linestyle=':')
    corr = df['prefix_ratio'].corr(df['scatter'])
    Ftest = np.square(corr) / (1 - np.square(corr)) * (df.shape[0] - 2)
    pval = max(1 - fdtr(1, df.shape[0] - 2, Ftest), 0.001)
    axs[0, 1].annotate(f'$r={corr:.3f}$\n$p{"<" if pval == 0.001 else "="}{pval:.3f}$',
                       (0.82, 0.9), xycoords='axes fraction')
    # 轨迹长度对分布模式的影响
    cut_range = np.linspace(0, 22, 12)
    df['cut'] = pd.cut(df['trip_len'], cut_range,
                       labels=np.round((cut_range[:-1] + cut_range[1:]) / 2).astype(int))
    sb.pointplot(data=df, x='cut', y='scatter', estimator='median',
                 errorbar=('pi', 50), ax=axs[1, 0])
    axs[1, 0].set_title('(c)')
    axs[1, 0].set_xlabel('distance from origin to destination (km)')
    axs[1, 0].set_ylabel('standard distance (km)')
    axs[1, 0].set_xticks(np.arange(cut_range.shape[0]) - 0.5, cut_range.astype(int))
    axs[1, 0].grid(linestyle=':')
    corr = df['trip_len'].corr(df['scatter'])
    Ftest = np.square(corr) / (1 - np.square(corr)) * (df.shape[0] - 2)
    pval = max(1 - fdtr(1, df.shape[0] - 2, Ftest), 0.001)
    axs[1, 0].annotate(f'$r={corr:.3f}$\n$p{"<" if pval == 0.001 else "="}{pval:.3f}$',
                       (0.02, 0.9), xycoords='axes fraction')
    # 出发时间对分布模式的影响
    sb.pointplot(data=df, x='start_h', y='scatter', estimator='median',
                 errorbar=('pi', 50), ax=axs[1, 1])
    axs[1, 1].set_title('(d)')
    axs[1, 1].set_xlabel('departure time in a day (hour)')
    axs[1, 1].set_ylabel('standard distance (km)')
    axs[1, 1].grid(linestyle=':')
    axs[1, 1].set_xticks(np.linspace(0, 24, 7) - 0.5, np.linspace(0, 24, 7, dtype=int))
    # 日期对预测精度的影响
    # sb.pointplot(data=df, x='weekday', y='err', hue='model', estimator='median',
    #              errorbar=None, dodge=False, ax=axs[1, 1])
    # axs[1, 1].set_ylabel('median of prediction error (km)')
    # axs[1, 1].grid(linestyle=':')
    # axs[1, 1].set_xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.savefig(TRAJ_ROOT / 'figures/figure8.png')
    plt.show()


def draw_spatial_stat_tif(npz_file, metadata, column, output_tif, px_size=50, radius=500):
    """
    统计研究区内不同区域的预测指标：目的地预测误差，目的地的分布标准距离。
    平滑方式：radius半径的高斯核函数，进行数值平滑
    输出到tif栅格
    """
    npz = np.load(npz_file)
    err = _xy_distance(torch.as_tensor(npz['y_pred']),
                       torch.as_tensor(npz['y_true'])).numpy()
    scatter = np.sqrt(np.var(npz['p_sample'], axis=1).sum(axis=-1)) / _resolution / 1000
    df = pd.DataFrame({'err': err / _resolution / 1000, 'err_rel': err / npz['others'][:, 0],
                       'trace_id': npz['others'][:, 1].astype(np.int_),
                       'prefix_len': npz['others'][:, 2].astype(np.int_),
                       'scatter': scatter})
    meta = pd.read_csv(metadata, usecols=['trace_id', 'num_record', 'dest_x', 'dest_y', 'trip_len'])
    df = df.merge(meta, on='trace_id')
    df['prefix_ratio'] = df['prefix_len'] / df['num_record']
    df['trip_len'] = df['trip_len'] / _resolution / 1000
    geop = mp1_to_geopoint(df[['dest_x', 'dest_y']].to_numpy())
    # points = [geo.Point(geop[i, 0], geop[i, 1]) for i in range(geop.shape[0])]
    # df = gpd.GeoDataFrame(df, geometry=points, crs=PROJ_EPSG)
    # df.to_file(output_shp)
    kdtree = KDTree(geop)
    xs = np.arange(RESEARCH_BOUND[0], RESEARCH_BOUND[2] + px_size, px_size)
    ys = np.arange(RESEARCH_BOUND[3], RESEARCH_BOUND[1] - px_size, -px_size)
    xy = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    idx, distance = kdtree.query_radius(xy, r=radius * 2, return_distance=True, sort_results=True)
    result = jl.Parallel(8)(jl.delayed(_spatial_gaussian_kernel)(df[column], _idx, _dist, radius)
                            for _idx, _dist in zip(idx, distance))
    raster = np.empty((xs.shape[0], ys.shape[0]), dtype=np.float32)
    xs = np.arange(xs.shape[0])
    ys = np.arange(ys.shape[0])
    xy = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    raster[xy[:, 0], xy[:, 1]] = result
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(str(output_tif), xsize=raster.shape[0], ysize=raster.shape[1],
                           bands=1, eType=gdal.GDT_Float32)
    dst_ds.SetGeoTransform([RESEARCH_BOUND[0], px_size, 0, RESEARCH_BOUND[3], 0, -px_size])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(PROJ_EPSG.split(':')[-1]))
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(raster.T)


def _spatial_gaussian_kernel(df, idx, distance, bandwidth, min_sample=5):
    if idx.shape[0] < min_sample or \
            np.count_nonzero(distance < bandwidth) < min_sample:
        return np.nan
    weight = np.exp(np.square(distance / bandwidth) * -3)
    return np.average(df.iloc[idx], weights=weight)


if __name__ == '__main__':
    # generate_road_heatmap()
    # generate_timeseries_plot()
    # cal_distance_stat()
    # corresp = []
    # shps = []
    # for code, region, pinyin in [('04', '福田', 'ft'), ('05', '南山', 'ns')]:
    #     roads1 = pd.read_csv(ROAD_SEG_ROOT / f'{region}区/correspondence_{pinyin}_17q4.txt', header=None)
    #     roads1.columns = correspondence_header
    #     roads2 = gpd.read_file(ROAD_SEG_ROOT / f'{region}区/RoadSegment_17q4.TAB')
    #     corresp.append(roads1)
    #     shps.append(roads2)
    # corresp = pd.concat(corresp)
    # shps = gpd.GeoDataFrame(pd.concat(shps))
    # print(shps.columns)
    # print(corresp.shape[0], corresp['link_id'].nunique())
    # print(len([road for road, _ in corresp.groupby(['mesh_code', 'road_id'])]))
    # print(shps.shape[0], len([road for road, _ in shps.groupby(['Mesh_Code', 'ROAD_ID'])]))

    # file = np.load(TRAJ_ROOT / 'embedding/POIemb_500m_k10_rel_torch.npz')
    # vector_look(file['emb_center'], title_scatter='POI embeddings by PCA', metric='cosine',
    #             corr_mat=True, title_corr='correlation matrix of POI types', items=file['poiType'])
    # array = np.load(TRAJ_ROOT / 'embedding/roademb_vgae_gat_d32_neg10_ep500.npy')
    # print(array.shape)
    # vector_look(array, title_scatter='road embeddings', metric='pca',
    #             pca_eigen=True, title_eigen='PCA explained variance ratio of road embedding')
    # embedding_3dcolor(array, output=TRAJ_ROOT / 'analyze/road1_3dcolor_gat_neg10.csv')

    # draw_loss_curve(TRAJ_ROOT / 'predict_model/transformer1.csv',
    #                 title='training loss', xlabel='epoch',
    #                 ylabel=['loss (total, cross entropy)', 'loss (time mse)'],
    #                 twinx=['time_mse'])

    # stat_label_distribution(TRAJ_ROOT / 'trajectories/inputs/metadata.csv',
    #                         TRAJ_ROOT / 'analyze/dest_road_count.csv',
    #                         title='cumulate histogram of trajectories w.r.t. top K frequent labels',
    #                         xlabel='top K frequent labels', ylabel='proportion of trajectories',
    #                         subplot_xmax=1000)

    # draw_err_hist([TRAJ_ROOT / 'analyze/err_distrib_transformer2_compress.npz',
    #                TRAJ_ROOT / 'analyze/err_distrib_transformer5_interval.npz'],
    #               ['distance_err'] * 2, range=(0, 10000),
    #               title='distance error distribution (m)', xlabel='distance to target (m)',
    #               legend=['classification', 'regression'])
    # sample_pred_distribution(TRAJ_ROOT / 'trajectories/inputs', TRAJ_ROOT / 'roads/road_input.csv',
    #                          TRAJ_ROOT / 'predict_model/diffusion3_all_feat.pth',
    #                          TRAJ_ROOT / 'analyze/prediction_diffusion3.shp',
    #                          test_fn='p_sample', trace_id=526398,
    #                          proportions=[0.3, 0.6, 0.82, 0.95],
    #                          # trace_id=141, 54994, 55042, 526398, 76673, 288211
    #                          )
    # analyze_pred_err_distr({'Diffusion model': TRAJ_ROOT / 'analyze/diffusion3_all_feat_psample.npz',
    #                         'Trajectory encoder': TRAJ_ROOT / 'analyze/transformer5_poiall3_result.npz'},
    #                        TRAJ_ROOT / 'trajectories/inputs/metadata.csv')
    analyze_psample_scatter(TRAJ_ROOT / 'analyze/diffusion3_all_feat_psample.npz',
                            TRAJ_ROOT / 'trajectories/inputs/metadata.csv')
    # draw_spatial_stat_tif(TRAJ_ROOT / 'analyze/diffusion3_all_feat_psample.npz',
    #                       TRAJ_ROOT / 'trajectories/inputs/metadata.csv', 'scatter',
    #                       TRAJ_ROOT / 'analyze/prediction_scatter_spatial.tif')
