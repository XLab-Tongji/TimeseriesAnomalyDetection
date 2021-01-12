import os
import torch
from torch import device
import glob
import datetime
import numpy as np
import shutil
from pathlib import Path
import pickle

def normalization(seqData,max,min):
    return (seqData -min)/(max-min)

def standardization(seqData,mean,std):
    return (seqData-mean)/std

def reconstruct(seqData,mean,std):
    return seqData*std+mean

class PickleDataLoad(object):
    def __init__(self, data_type, filename, augment_test_data=True):
        self.augment_test_data=augment_test_data
        self.trainData, self.trainLabel = self.preprocessing(Path('dataset',data_type,'labeled','train',filename),train=True)
        self.testData, self.testLabel = self.preprocessing(Path('dataset',data_type,'labeled','test',filename),train=False)

     #增强的主要操作是引入噪声
    def augmentation(self,data,label,noise_ratio=0.05,noise_interval=0.0005,max_length=100000):
        noiseSeq = torch.randn(data.size())   #产生一个和原数据大小相同的标准正态分布中的一组随机数
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            #生成引入的噪声
            scaled_noiseSeq = noise_ratio * self.std.expand_as(data) * noiseSeq
            #原数据和引入噪声后的数据合并，标签不改变
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
            augmentedLabel = torch.cat([augmentedLabel, label])
            #加上多组这样的噪声数据，控制最大长度
            if len(augmentedData) > max_length:
                augmentedData = augmentedData[:max_length]
                augmentedLabel = augmentedLabel[:max_length]
                break

        return augmentedData, augmentedLabel

    def preprocessing(self, path, train=True):
        """ Read, Standardize, Augment """

        with open(str(path), 'rb') as f:
            data = torch.FloatTensor(pickle.load(f))  #将数据转化成张量
            label = data[:,-1]   #复制最后一列的标签
            data = data[:,:-1]   #复制除最后一列的前几列数据值
        if train:
            self.mean = data.mean(dim=0)   #计算每列平均值
            self.std= data.std(dim=0)      #计算每列标准差
            self.length = len(data)
            data,label = self.augmentation(data,label)
        else:
            if self.augment_test_data:
                data, label = self.augmentation(data, label)  #测试集没有计算均值和标准差的步骤，所以引入的噪声为0

        data = standardization(data,self.mean,self.std)  #标准化

        return data,label

    def batchify(self,args,data, bsz):
        nbatch = data.size(0) // bsz  #整数除
        trimmed_data = data.narrow(0,0,nbatch * bsz) #限制为nbatch * bsz大小的数据
        #先转置而后把tensor变成在内存中连续分布的形式
        batched_data = trimmed_data.contiguous().view(bsz, -1, trimmed_data.size(-1)).transpose(0,1)
        #把tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
        batched_data = batched_data.to(device(args.device))
        return batched_data

