# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:40:31 2020

@author: yuanyuan
"""
import os
import tensorflow as tf
import numpy as np
from configurationtest import clstm_config
from convlstm_classifiertest import clstm_clf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python import pywrap_tensorflow as pt


outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", "1605440722"))
reader = pt.NewCheckpointReader(os.path.join(outdir, 'model.ckpt'))


# 获取 变量名: 形状
vars = reader.get_variable_to_shape_map()
for k in sorted(vars):
    print(k, vars[k])
saver=tf.train.import_meta_graph(os.path.join(outdir, 'model.ckpt.meta'),clear_devices=True)

with tf.Session(graph=tf.get_default_graph()) as sess:
   
    # 序列化模型
    input_graph_def = sess.graph.as_graph_def()
    # 2. 载入权重
    saver.restore(sess, os.path.join(outdir, 'model.ckpt'))
    
   
    # 3. 转换变量为常量
    sess.run(tf.global_variables_initializer())
    output_graph_def =tf.graph_util.convert_variables_to_constants(sess, input_graph_def,output_nodes)
    # 4. 写入文件
    with open(os.path.join(outdir, 'frozen_model.pb'), "wb") as f:
        f.write(output_graph_def.SerializeToString())


    
