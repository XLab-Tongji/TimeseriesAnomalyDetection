from dataset.tf_dataset import get_tf_dataset
from model.supervised.models import get_cnn
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

_,ds = get_tf_dataset()
path = os.path.join("tmp","supervised")
model = tf.keras.models.load_model(path)

total = 0
succes = 0
fp = 0
fn = 0
tp = 0
tn = 0

for x,y in tqdm(ds,desc="evaluating"):
    y_pred = model(x)
    for i in range(y.shape[0]):
        total+=1
        #print("y_true:{},y_pred:{}".format(y[i],y_pred[i]))
        l_pred = np.argmax(y[i])
        l_true = np.argmax(y_pred[i])
        if l_pred==l_true:
            succes+=1
            if l_pred==0:
                tp+=1
            else:
                tn+=1
        else:
            if l_pred==0:
                fp+=1
            else:
                fn+=1
        
print("#################################")
accuracy = succes/total
precision = tp/(tp+fp)
recall=tp/(tp+fn)
print("accuracy:{}\nprecision:{}\nrecall:{}".format(accuracy,precision,recall))
print("#################################")