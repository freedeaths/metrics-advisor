import os, shutil, glob
import time, tarfile, uuid, datetime

from mathbox.statistics.estimator import ncc
#from mathbox.app.signal.correlation import max_corr
#from mathbox.app.signal.change_point import e_divisive
from mathbox.app.signal.filter import moving_median, f_lowpass_filter
from mathbox.app.signal.outlier import noise_outlier
#from energy_statistics import e_divisive # pure python
from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive # with c lib

import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.transforms as mtrans
import pandas as pd
from jinja2 import PackageLoader,Environment


def get_valid_signals(path):
    csv_files = glob.glob(path)
    signals = []
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

    tmp_dir = './' + str(uuid.uuid4()) + '/'
    report_path = './reports/'
    try:
        os.mkdir(report_path)
    except:
        pass
    tar = tarfile.open('./metrics/write-auto-inc-rand-batch-point-get.tar.gz')
    #tar = tarfile.open('./metrics/write-auto-inc-full-index-lookup.tar.gz')
    #tar = tarfile.open('./metrics/rand-batch-point-get.tar.gz')
    #tar = tarfile.open('./metrics/write-auto-inc.tar.gz')
    #tar = tarfile.open('./metrics/fix-update-key.tar.gz')
    #tar = tarfile.open('./metrics/full-index-lookup.tar.gz')
    files = [file for file in tar.getmembers() if file.name.endswith('.csv')]
    head, _ = os.path.split(files[0].name)
    tar.extractall(tmp_dir, files)
    tar.close()
    signals = get_valid_signals(tmp_dir + head + '/*.csv')
    
    count_bucket = 40 # 15 seconds * 40 = 10 minutes
    sample_time_step = 15 # seconds
    buckets = []

    time_min, time_max = time_minmax(signals)
    samples = (time_max - time_min) // sample_time_step + 1 # 480
    
    for i in range(samples//count_bucket + 1):
        buckets.append({'start':time_min + i*count_bucket*sample_time_step, 'obj':[], 'candidates': []})

    start = time.time()
    
    # set T
    #obj_signals = ['tidb_p99_rt:total','tidb_p99_get_token_dur','tidb_conn_cnt:by_instance','tidb_heap_size:by_instance']
    obj_signals = ['tidb_p99_rt:total','tidb_p99_get_token_dur','tidb_heap_size:by_instance']

    # set B to different time buckets
    for item in signals:
        if max(item['data'].tolist()) - min(item['data'].tolist()) > 0.005:
            med_filtered = moving_median(item['data'].tolist(), 5) # filtered
            cp = e_divisive(med_filtered,pvalue=0.05,permutations=100) # filtered
            #cp = e_divisive(item['data'].tolist(),pvalue=0.05,permutations=100) # original
            outlier = get_noise(item['data'].tolist(), 5, sample_time_step, 3, 0.01/sample_time_step, 3)
        
            anomaly = list(set(cp + outlier)) # 暂时不区分异常和变化点

            if anomaly != []:
                anomaly_timestamps = [item['timestamp'][x] for x in cp]

                for i in anomaly_timestamps:
                    num = (i-time_min)//sample_time_step//count_bucket
                    if item not in buckets[num]['obj'] and item not in buckets[num]['candidates']:
                        if item['name'] in obj_signals:
                            buckets[num]['obj'].append(item)
                        else:
                            buckets[num]['candidates'].append(item)
    end = time.time()
    print("CPD and outlier time cost: ", end-start)
    start = time.time()

    suffix = str(tmp_dir)[-9:-1]

    stats = []
    # correlation in each bucket
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
                plt.plot(datenums,obj_data,'ro-', label=obj['name'], alpha=0.5)
                plt.plot(datenums,moving_median(obj_data, 5),'bo-', label=obj['name']+'_filtered', alpha=0.3) # plot filtered data
                
                for it in sort_corr[:3]:
                    can = next(item for item in bucket['candidates'] if item['name'] == it['candidate']['name'] and item['node'] == it['candidate']['node'])
                    can_data = get_relative(can['data'].tolist()[40*i:40*i+40])
                    #plt.plot(can['timestamp'][40*i:40*i+40].tolist(),can_data, label='max '+str(sort_corr.index(it)+1)+' '+can['name']+'__'+can['node'])
                    datenum = md.date2num([datetime.datetime.fromtimestamp(ts) for ts in can['timestamp'][40*i:40*i+40].tolist()])
                    tr = mtrans.offset_copy(plt.gca().transData, fig=plt.gcf(), x=0.0, y=-1.5, units='points')
                    plt.plot(datenum,can_data, label='top '+str(sort_corr.index(it)+1)+' '+can['name']+'__'+can['node'], alpha=0.5, transform=tr)
                plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
                plt.legend(framealpha=0.3, fontsize=8)
                plt.xticks(rotation=45, fontsize=8)
                plt.savefig('./reports/bucket_'+str(i)+'_'+obj['name']+'_'+ suffix +'.png')
                plt.close()
                plt.cla()
                plt.clf()
        stats.append(cor)

    end = time.time() # about 320s
    print("Correlation time cost: ", end-start)

    shutil.rmtree(tmp_dir)

    foo = {'bar':'metrics-advisor team'}
    
    pics = [f for f in os.listdir(report_path) if f.endswith(suffix+ '.png')]

    dict = {'foo':foo,'anomaly':buckets,'sort_corr':stats, 'pics':pics}
    env = Environment(loader=PackageLoader('metrics_advisor','templates'))
    template = env.get_template('report.tpl')
    output = './reports/report_'+suffix+'.md'
    with open(output,'w') as f:
        f.write(template.render(dict))