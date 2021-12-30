#from mathbox.app.signal.correlation import max_corr
import matplotlib.pyplot as plt
import glob
import os
from numpy import sign
import pandas as pd
import time

from mathbox.statistics.estimator import ncc
#from mathbox.app.signal.change_point import e_divisive
from mathbox.app.signal.filter import moving_median, f_lowpass_filter
from mathbox.app.signal.outlier import noise_outlier

from energy_statistics import e_divisive

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

    # 目前的问题：
    # 1. 时间戳没有对齐，应该要对齐，暂时取交集
    # 2. NaN/Null/None 的处理暂时交由上游处理，一是零星的插值，二是大片没有的删除，三是有 Trigger 的，应该补，暂时先直接删除 NaN 行数占比来处理。
    #signals = get_valid_signals("/Users/sunyishen/PingCAP/repos/playground/prom_metrics/metrics-analysis-data/full-index-lookup/reshape/*.csv") # 整个，约400秒
    signals = get_valid_signals("/Users/sunyishen/PingCAP/repos/playground/metrics-advisor/metrics/rand-batch-point-get/*.csv")
    #signals = get_valid_signals("/Users/sunyishen/PingCAP/repos/playground/prom_metrics/metrics-analysis-data/full-index-lookup/reshape/node_disk_write_dur:by_instance:by_device.csv") # 单个，省时间

    count_bucket = 40 # 15 seconds * 40 = 10 minutes
    sample_time_step = 15 # seconds
    buckets = []
    change_points = []
    outliers = []
    detected_signals = []

    time_min, time_max = time_minmax(signals)
    samples = (time_max - time_min) // sample_time_step + 1 # 480
    
    for i in range(samples//count_bucket):
        #buckets.append({'start':time_min + i*count_bucket*sample_time_step, 'end':time_min + (i+1)*count_bucket*sample_time_step-sample_time_step, 'count':0})
        buckets.append({'start':time_min + i*count_bucket*sample_time_step, 'obj':[], 'candidates': []})

    start = time.time()
    

    # obj_signals = ['tidb_p99_rt:total','tidb_p99_get_token_dur','tidb_conn_cnt:by_instance','tidb_heap_size:by_instance']
    obj_signals = ['tidb_p99_rt:total']

    # main loop
    for item in signals:
        # item-file, item[1]-df, item[1].columns]
        cp = e_divisive(item['data'].tolist(),pvalue=0.05,permutations=100) # return an index list
        outlier = get_noise(item['data'].tolist(), 5, sample_time_step, 3, 0.01/sample_time_step, 3) # return an index list

        anormaly = list(set(cp + outlier)) # 暂时不区分异常和变化点

        if anormaly != []:
            anormaly_timestamps = [item['timestamp'][x] for x in cp]

            for i in anormaly_timestamps:
                if item not in buckets[(i-time_min)//sample_time_step//count_bucket]['obj'] and item not in buckets[(i-time_min)//sample_time_step//count_bucket]['candidates']:
                    if item['name'] in obj_signals:
                        buckets[(i-time_min)//sample_time_step//count_bucket]['obj'].append(item)
                    else:
                        buckets[(i-time_min)//sample_time_step//count_bucket]['candidates'].append(item)


    for bucket in buckets:
        correlation = []
        i = buckets.index(bucket)
        if bucket['obj'] != []: # 暂时没循环obj
            for candidate in bucket['candidates']:
                a = bucket['obj'][0]['data'][40*i:40*i+40].tolist()
                b = candidate['data'][40*i:40*i+40].tolist()
                tmp = ncc(a, b, lag_max=10)
                corr = max(tmp, key=lambda x:x[1])
                correlation.append({'name': candidate['name'],'corr': corr})
            sort_corr = sorted(correlation, key=lambda x:x['corr'][1], reverse=True)
            print(sort_corr)
            obj_data = get_relative(bucket['obj'][0]['data'].tolist()[40*i:40*i+40])
            plt.plot(obj_data,'ro-', label=bucket['obj'][0]['name'])
            for it in sort_corr[:5]:
                can = next(item for item in bucket['candidates'] if item['name'] == it['name'])
                can_data = get_relative(can['data'].tolist()[40*i:40*i+40])
                plt.plot(can_data, label='max '+str(sort_corr.index(it)+1)+' '+can['name'])
            plt.legend()
            #plt.show()
            plt.savefig('/Users/sunyishen/PingCAP/repos/playground/metrics-advisor/reports/bucket_'+str(i)+'.png')


    #correlation = max_corr(obj, candidates, 4)[1:]
    #print("correlation is: {}".format(correlation))
    end = time.time() # about 320s
    print("CPD time cost: ", end-start)


 