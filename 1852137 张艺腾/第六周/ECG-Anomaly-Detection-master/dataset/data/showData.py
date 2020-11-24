import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt

#load normal and abnormal egc from csv files
normal = np.array(pd.read_csv("/Users/kerrzhang/Downloads/ECG-Anomaly-Detection-master/dataset/data/normal.csv"));
abnormal = np.array(pd.read_csv("/Users/kerrzhang/Downloads/ECG-Anomaly-Detection-master/dataset/data/abnormal.csv"));

x=range(0,188);
plt.plot(x,abnormal[2]);
plt.show();