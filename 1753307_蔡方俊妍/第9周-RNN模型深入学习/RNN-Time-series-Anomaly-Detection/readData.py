# -*- coding: utf-8 -*-

import  pickle
from pathlib import Path
import os


# 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
def getRNN(dataname,model):
    #dataname='power_demand'
    #filename='power_data'
        #参数1 dataname filename
    if dataname=='respiration':
        filename='nprs44'
    elif dataname=='power_demand':
        filename='power_data'
    
    print("========================================================")
    ss=str(Path('dataset',dataname,'labeled','test',filename).with_suffix('.pkl'))
    f=open(ss,'rb')
    data = pickle.load(f)
    datas=[]
    anomalys=[]
    for i in range(len(data)):
        point=[]
        point.append(i)
        point.append(data[i][0])
        datas.append(point)
        if data[i][1]==1:
            anomalys.append(i)
    print(anomalys)
    #print(len(anomalys))
    
    #参数2 使用的模型model
    #model='LSTM'
    command='python 2_anomaly_detection.py --data '+dataname+' --filename '+filename+' --model '+model
    #os.system(command)
    
    print("========================================================")
    ss=str(Path('result',dataname,filename,model,'score').with_suffix('.pkl'))
    f=open(ss,'rb')
    anomaly_score = pickle.load(f)
    #单个数据则用item将tensor转换为值,数组则用tolist
    anomaly_score=anomaly_score[0].tolist()
    scores=[]
    for i in range(len(anomaly_score)):
        point=[]
        point.append(i)
        point.append(anomaly_score[i])
        scores.append(point)
    
    
    #print(scores)
    return datas,anomalys,scores





