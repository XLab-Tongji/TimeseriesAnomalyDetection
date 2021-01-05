# -*- coding: utf-8 -*-

import numpy
import pickle
from pathlib import Path
import os


# 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
def getRNN(dataname, model):

    # dataname='power_demand'
    # filename='power_data'
    # 参数1 dataname filename
    if dataname == 'respiration':
        filename = 'nprs44'
    elif dataname == 'power_demand':
        filename = 'power_data'

    print("========================================================")
    print(dataname)
    print(model)
    print("iiiiiiiiiiiiiiiiipppppppppppppppppppppp")
    ss = str(
        Path('./TimeseriesAnomalyDetection\\RNN\\dataset', dataname, 'labeled', 'test', filename).with_suffix('.pkl'))
    f = open(ss, 'rb')
    data = pickle.load(f)
    datas = []
    anomalys = []
    for i in range(len(data)):
        point = []
        point.append(i)
        point.append(data[i][0])
        datas.append(point)
        if data[i][1] == 1:
            anomalys.append(i)
    print(anomalys)
    # print(len(anomalys))

    # 参数2 使用的模型model
    # model='LSTM'
    command = 'python .\\TimeseriesAnomalyDetection\\RNN\\2_anomaly_detection.py --data ' + dataname + ' --filename ' + filename + ' --model ' + model
    # 如果需要在线预测，则取消下面一行代码的注释（在线预测耗时较久）
    # os.system(command)

    print("========================================================")
    ss = str(Path('.\\TimeseriesAnomalyDetection\\RNN\\result', dataname, filename, model, 'score').with_suffix('.pkl'))
    f = open(ss, 'rb')
    anomaly_score = pickle.load(f)
    # 单个数据则用item将tensor转换为值,数组则用tolist
    anomaly_score = anomaly_score[0].tolist()
    scores = []
    for i in range(len(anomaly_score)):
        point = []
        point.append(i)
        point.append(anomaly_score[i])
        scores.append(point)

    # print(scores)
    return datas, anomalys, scores


def takeSecond(elem):
    return elem[1]


def get_RNN_result(dataname, model):
    datas, anomalys, scores = getRNN(dataname, model)
    data_len = len(datas)
    anomaly_len = len(anomalys)
    score_list = []
    is_anomaly = numpy.zeros((data_len,), dtype=int)
    for i in range(0, anomaly_len):
        is_anomaly[anomalys[i]] = 1
    for i in range(0, data_len):
        score_list.append((is_anomaly[i], scores[i][1], i))
    score_list.sort(key=takeSecond, reverse=True)
    res = -anomaly_len*(anomaly_len + 1)
    for i in range(0, data_len):
        res += score_list[i][0] * (data_len - i)
    res /= anomaly_len*(data_len - anomaly_len)
    if res < 0.5:
        res = 1 - res
    print("auc_res=", res)
    anomaly_ranks = []
    lastl = 0
    lastr = -1
    for i in range(0, data_len):
        if is_anomaly[i] == 1:
            if lastr == -1:
                lastl = i
            lastr = i
        elif lastr != -1:
            anomaly_ranks.append([lastl, lastr])
            lastr = -1
    de_anomaly = numpy.zeros((data_len,), dtype=int)
    for i in range(0, anomaly_len):
        de_anomaly[score_list[i][2]] = 1
    detect_res = [[] for i in range(2)]
    for i in range(data_len):
        detect_res[int(de_anomaly[i])].append(datas[i])
        detect_res[1 - int(de_anomaly[i])].append('-')
        if i > 0 & (int(de_anomaly[i]) != int(de_anomaly[i-1])):
            detect_res[int(de_anomaly[i])][i - 1] = datas[i - 1]
    return datas, anomaly_ranks, detect_res, res

