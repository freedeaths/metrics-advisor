import json
import os, shutil, glob
import time, tarfile, uuid, datetime
#from mathbox.app.signal.correlation import max_corr
import sys
from typing import List, Dict, Any

from jinja2 import PackageLoader,Environment
import tarfile
import shutil
import uuid
import argparse
import tempfile

import glob
import os
import time

from mathbox.statistics.estimator import ncc
#from mathbox.app.signal.correlation import max_corr
#from mathbox.app.signal.change_point import e_divisive
from mathbox.app.signal.filter import moving_median, f_lowpass_filter
from mathbox.app.signal.outlier import noise_outlier
#from energy_statistics import e_divisive # pure python
from numpy import ndarray
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.core.generic import NDFrame
from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive # with c lib

import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.transforms as mtrans
import pandas as pd
from jinja2 import PackageLoader, Environment
from alive_progress import alive_bar
import networkx as nx


def get_valid_signals(path):
    csv_files = glob.glob(path)
    signals: list[dict[str, Series | ExtensionArray | None | ndarray | DataFrame | NDFrame | Any]] = []
    for f in csv_files:
        data = pd.read_csv(f)
        if data.shape[0] > 20:
            name = os.path.splitext(os.path.basename(f))[0]
            for col in data.columns:
                if col != 'timestamp':
                    signals.append({'name':name, 'node':col, 'timestamp':data['timestamp'], 'data':data[col]})
    return signals


def get_noise(data, window, T, n, f_min, level):
    med_filtered = moving_median(data, window)

    moved_trend = [x - y for x, y in zip(data, med_filtered)]

    _, seasonality = f_lowpass_filter(moved_trend, T, n, f_min)

    moved_seasonality = [x - y for x, y in zip(moved_trend, seasonality)]

    outlier = noise_outlier(moved_seasonality, level)

    return outlier


def time_minmax(data): # 为了解耦牺牲了时间，放到 get_valid_signals 中可以省点时间
    time_min = 1e32
    time_max = 0
    for item in data:
        time_min_tmp = item['timestamp'].min()
        time_max_tmp = item['timestamp'].max()
        if time_min_tmp < time_min:
            time_min = time_min_tmp
        if time_max_tmp > time_max:
            time_max = time_max_tmp
    return time_min, time_max


def get_relative(list):
    max_element = abs(max(list)) + 1e-10
    return [x / max_element for x in list]


if __name__ == "__main__":
    # 目前：
    # TODO: 手动去掉所有不连续的时间
    # TODO: 修 BUG seek_ops 的相关性对不对
    # 1. 时间暂时没有对齐，因为相关性有 lag 参数
    # 2. NaN/Null/None 的处理暂时交由上游处理，一是零星的插值，二是大片没有的删除，三是有 Trigger 的，应该补，暂时先直接删除 NaN 行数占比来处理。

    parser = argparse.ArgumentParser(description="""
        metrics_advisor.py detect interval with abnormal points and find the most relate metrics""")
    parser.add_argument('-i', '--input', dest='input', help='input tar file',
                        required=True)
    parser.add_argument('-o', '--output', dest='output', help='output dir', required=False, default='./reports/')
    parser.add_argument('--top', dest='top', help='set topN', required=False, default=3)
    parser.add_argument('-vv', '--verbose', dest='verbose', type=bool, default=False, help='if verbose is set, show detail in result table')

    args = parser.parse_args()
    input_tar = args.input
    output_dir = args.output
    top_n = int(args.top)
    verbose = args.verbose

    sys_tmp = tempfile.gettempdir()
    tmp_dir = os.path.join(sys_tmp, "metrics-advisor", str(uuid.uuid4()))
    # tmp_dir = sys_tmp + str(uuid.uuid4()) + '/'
    # report_path = './reports/'
    try:
        os.mkdir(output_dir)
    except:
        pass
    #tar = tarfile.open('./metrics/write-auto-inc-rand-batch-point-get.tar.gz')
    #tar = tarfile.open('./metrics/write-auto-inc-full-index-lookup.tar.gz')
    #tar = tarfile.open('./metrics/rand-batch-point-get.tar.gz')
    #tar = tarfile.open('./metrics/write-auto-inc.tar.gz')
    #tar = tarfile.open('./metrics/fix-update-key.tar.gz')
    #tar = tarfile.open('./metrics/full-index-lookup.tar.gz')
    # tar = tarfile.open('./metrics/cluster-4048.gz.tar')
    tar = tarfile.open(input_tar)
    files = [file for file in tar.getmembers() if file.name.endswith('.csv')]
    head, _ = os.path.split(files[0].name)
    tar.extractall(tmp_dir, files)
    tar.close()

    signals = get_valid_signals( os.path.join(tmp_dir,head, '*.csv'))
    # signals = get_valid_signals("/Users/sunyishen/PingCAP/repos/playground/prom_metrics/metrics-analysis-data/full-index-lookup/reshape/node_disk_write_dur:by_instance:by_device.csv") # 单个，省时间

    count_bucket = 40 # 15 seconds * 40 = 10 minutes
    sample_time_step = 15 # seconds
    buckets = []

    time_min, time_max = time_minmax(signals)

    print('time_min:{0}({1}), time_max:{2}({3}) having {4} seconds'.format(
        time_min, datetime.datetime.utcfromtimestamp(time_min),
        time_max, datetime.datetime.utcfromtimestamp(time_max),
        time_max - time_min))
    # print('time_min:', time_min, 'time_max:', time_max, 'duration:', time_max - time_min)
    samples = (time_max - time_min) // sample_time_step + 1 # 480
    for i in range(samples//count_bucket + 1):
        buckets.append({'start': time_min + i*count_bucket*sample_time_step, 'obj': [], 'candidates': [], 'anomaly_ts': []})

    start = time.time()
    
    # set T
    #obj_signals = ['tidb_p99_rt:total','tidb_p99_get_token_dur','tidb_conn_cnt:by_instance','tidb_heap_size:by_instance']
    # obj_signals = ['tidb_p99_rt:total', 'tidb_p99_get_token_dur', 'tidb_mem_usage']
    obj_signals = ['tidb_p99_rt:total']
    # set B to different time buckets
    with alive_bar(len(signals), title='detecting anomaly', length=30) as single_progress_bar:
        for item in signals:
            if max(item['data'].tolist()) - min(item['data'].tolist()) > 0.005:
                med_filtered = moving_median(item['data'].tolist(), 5) # filtered
                cp = e_divisive(med_filtered,pvalue=0.05,permutations=100) # filtered
                #cp = e_divisive(item['data'].tolist(),pvalue=0.05,permutations=100) # original
                outlier = get_noise(item['data'].tolist(), 5, sample_time_step, 3, 0.01/sample_time_step, 3)
                # 暂时不区分异常和变化点
                anomaly = list(set(cp + outlier))

                if anomaly != []:
                    anomaly_timestamps = [item['timestamp'][x] for x in cp]

                    for i in anomaly_timestamps:
                        num = (i-time_min)//sample_time_step//count_bucket
                        if item not in buckets[num]['obj'] and item not in buckets[num]['candidates']:
                            if item['name'] in obj_signals:
                                buckets[num]['obj'].append(item)
                                buckets[num]['anomaly_ts'].append(md.date2num(datetime.datetime.fromtimestamp(i)))
                            else:
                                buckets[num]['candidates'].append(item)

            # update progress bar
            single_progress_bar()

    end = time.time()
    # print("CPD and outlier time cost: ", end-start)
    start = time.time()

    suffix = str(tmp_dir)[-9:-1]
    stats = []
    # correlation in each bucket
    pics = []
    with alive_bar(len(buckets), title='calculating correlation', length=30) as buckets_progress_bar:
        for bucket in buckets:
            correlation = []
            cor = []
            i = buckets.index(bucket)
            if bucket['obj'] != []:
                for obj in bucket['obj']:
                    for candidate in bucket['candidates']:
                        if max(candidate['data'].tolist()) - min(candidate['data'].tolist()) > 0.005:
                            a = obj['data'][40*i:40*i+40].tolist()
                            b = candidate['data'][40*i:40*i+40].tolist()
                            #if candidate['name'] == 'tikv_seek_ops:by_type': # for test
                            #    print("a: ", a) # for test
                            #    print("b: ", b) # for test
                            #    print("node: ", candidate['node']) # for test
                            tmp = ncc(a, b, lag_max=3) # 原来是 10 个点，现在只用了 3 个点，系统惯性小
                            corr = max(tmp, key=lambda x:abs(x[1])) # abs 的话把负相关也算上了
                            correlation.append({'candidate': candidate,'corr': corr}) # 改一下 candidate['name'] -> candidate
                    if correlation != []:
                        sort_corr = sorted(correlation, key=lambda x:abs(x['corr'][1]), reverse=True) # abs 的话把负相关也算上了
                    cor.append({'name': obj['name'], 'corre': sort_corr})
                    obj_data = get_relative(obj['data'].tolist()[40*i:40*i+40])
                    datenums = md.date2num([datetime.datetime.fromtimestamp(ts) for ts in obj['timestamp'][40*i:40*i+40].tolist()])
                    # plot original data
                    plt.plot(datenums, obj_data, 'r.-', label=obj['name'], alpha=0.7)
                    # plot filtered data
                    plt.plot(datenums, moving_median(obj_data, 5), 'r*-', label=obj['name']+'_filtered', alpha=0.7)
                    # add line to anomaly time series
                    for ats in bucket['anomaly_ts']:
                        plt.axvline(x=ats, color='black', linestyle='--', linewidth=0.5)

                    # plot correlated data
                    for it in sort_corr[:top_n]:
                        can = next(item for item in bucket['candidates'] if item['name'] == it['candidate']['name'] and item['node'] == it['candidate']['node'])
                        can_data = get_relative(can['data'].tolist()[40*i:40*i+40])
                        #plt.plot(can['timestamp'][40*i:40*i+40].tolist(),can_data, label='max '+str(sort_corr.index(it)+1)+' '+can['name']+'__'+can['node'])
                        datenum = md.date2num([datetime.datetime.fromtimestamp(ts) for ts in can['timestamp'][40*i:40*i+40].tolist()])
                        tr = mtrans.offset_copy(plt.gca().transData, fig=plt.gcf(), x=0.0, y=-1.5, units='points')
                        plt.plot(datenum, can_data, label='top'+str(sort_corr.index(it)+1)+' '+can['name']+'__'+can['node'], alpha=0.5, transform=tr)

                    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
                    plt.legend(bbox_to_anchor=(0, 1.01), loc='lower left', borderaxespad=0, framealpha=0.3, fontsize=8)
                    plt.xticks(rotation=45, fontsize=8)
                    plt.ylabel('normalized value')
                    fig_name = 'bucket_{0}_{1}_{2}.png'.format(str(i), obj['name'], suffix)
                    pics.append('./' + fig_name)
                    plt.savefig(os.path.join(output_dir, fig_name), bbox_inches="tight")
                    plt.close()
                    plt.cla()
                    plt.clf()
            stats.append(cor)

            # update progress bar
            buckets_progress_bar()

    end = time.time() # about 320s
    # print("Correlation time cost: ", end-start)

    shutil.rmtree(tmp_dir)

    foo = {'bar': 'metrics-advisor team'}
    
    # pics = [f for f in os.listdir(report_path) if f.endswith(suffix+ '.png')]
    dict = {'foo': foo, 'anomaly': buckets, 'sort_corr': stats, 'pics': pics, 'top_n': top_n}

    env = Environment(loader=PackageLoader('metrics_advisor','templates'))
    template = env.get_template('report.tpl')
    output = os.path.join(output_dir, 'report_{0}.md'.format(suffix))
    debug_info_path = os.path.join(output_dir, 'debug_info_{0}.md'.format(suffix))
    with open(output, 'w') as f:
        f.write(template.render(dict))
    # with open(debug_info_path, 'w') as f:
    if verbose:
        # print correlation score
        for (stats_idx, item) in enumerate(stats):
            if len(item) > 0:
                for obj in item:
                    print("obj: {0} at stats {1}".format(obj['name'], stats_idx))
                    for can in obj['corre']:
                        if abs(can['corr'][1]) < 0.9:
                            continue
                        print("metrics: {0}_{1} corr: {2}".format(can['candidate']['name'], can['candidate']['node'], can['corr']))
        # draw network
        first_bucket_idx = 0
        for bucket in buckets:
            if bucket['obj']:
                first_bucket_idx = buckets.index(bucket)
                break
        source = []
        target = []
        weight = []
        obj = buckets[first_bucket_idx]['obj'][0]
        candidates = []
        for candidate in buckets[first_bucket_idx]['candidates']:
            if max(candidate['data'].tolist()) - min(candidate['data'].tolist()) > 0.005:
                candidates.append(candidate)

        candidates = buckets[first_bucket_idx]['candidates']
        while len(candidates) > 0:
            print(len(candidates))
            correlation = []
            skip_idx = []
            for (i, candidate) in enumerate(candidates):
                a = obj['data'][40 * first_bucket_idx: 40 * first_bucket_idx + 40].tolist()
                b = candidate['data'][40 * first_bucket_idx: 40 * first_bucket_idx + 40].tolist()
                result = ncc(a, b, lag_max=3)  # 原来是 10 个点，现在只用了 3 个点，系统惯性小
                corr = max(result, key=lambda x: abs(x[1]))  # abs 的话把负相关也算上了
                if abs(corr[1]) < 0.9:
                    skip_idx.append(i)
                    continue
                correlation.append({'candidate': candidate, 'corr': corr, 'idx':i})  # 改一下 candidate['name'] -> candidate
            if correlation != []:
                sort_corr = sorted(correlation, key=lambda x: abs(x['corr'][1]), reverse=True)
                n = 5 if len(correlation) > 5 else len(correlation)
                for i, scorr in enumerate(sort_corr[:n]):
                    source.append('{0}__{1}'.format(obj['name'], obj['node']))
                    target.append('{0}__{1}'.format(scorr['candidate']['name'], scorr['candidate']['node']))
                    if i == 0:
                        weight.append(2)
                        obj = scorr['candidate']
                    else:
                        weight.append(1)
                # update candidates
                for scorr in sort_corr[:n]:
                    candidates.pop(scorr['idx'])
            else:
                for i in skip_idx[::-1]:
                    candidates.pop(i)

        edges = pd.DataFrame({'source': source,
                              'target': target,
                              'width': weight})
        print(edges)
        GG = nx.from_pandas_edgelist(edges, edge_attr='width')
        # pos = nx.spring_layout(GG)
        #
        # nx.draw_networkx_edges(GG, pos, width=weight)
        nx.draw(GG, with_labels=True)
        plt.savefig(os.path.join(output_dir, 'network.png'))
    print('report is saved at {0}'.format(output))
    # print('debug info is saved at {0}'.format(debug_info_path))

