# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:40:31 2020

@author: yuanyuan
"""
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


from dataloadertest import DataLoader



sess = tf.Session()
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", "1605976772"))
# 读取模型并保存到序列化模型对象中
with open(os.path.join(outdir, 'frozen_model.pb'), "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    acc_tensor,predict_tensor = tf.import_graph_def(graph_def,return_elements=["Accuracy:0", "softmax/predictions:0"])
    b_tensor,x_tensor = tf.import_graph_def(graph_def,return_elements=["batch_size:0", "input_x:0"])
    y_tensor,k_tensor = tf.import_graph_def(graph_def,return_elements=["input_y:0", "keep_prob:0"])
data_loader = DataLoader()
x_test,y_test,dataset,originAnoma=data_loader.get_test_data()


sess.run(tf.global_variables_initializer())
anoma=[]

input_x= [[[0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2],
          [0.2]]]
input_y2=[1]
fetches = {'accuracy': acc_tensor,'predictions': predict_tensor}
feed_dict={b_tensor:1
       ,x_tensor:input_x,y_tensor:input_y2,k_tensor:1.0}
vars = sess.run(fetches, feed_dict)
accuracy = vars['accuracy']
predictions = vars['predictions']
print(predictions)
   # precision, recall, f1, _ = precision_recall_fscore_support(input_y, predictions, average = 'binary')
  
    #print(accuracy)
   # print(predictions)