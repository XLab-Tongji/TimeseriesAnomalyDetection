# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:00:34 2020

@author: yuanyuan
"""

import numpy as np
test=np.load('dataset/train/0/00e409ac-f6bf-42d0-8e0d-9c65db8d9af0.npy',encoding = "latin1")  #加载文件
print(test)
print(np.size(test,0))