import tensorflow as tf
from model.unsupervised.train import l_enc,l_con
from dataset.tf_dataset import get_tf_dataset,plot_sample
import numpy as np
from model.unsupervised.models import *
from tqdm import tqdm
import os

TRESHOLD = 0.2

def get_models():
    base_path = os.path.join("tmp","unsupervised","models")
    gen_enc = tf.keras.models.load_model(os.path.join(base_path,"gen_enc"))
    gen_dec = tf.keras.models.load_model(os.path.join(base_path,"gen_dec"))
    enc = tf.keras.models.load_model(os.path.join(base_path,"enc"))
    return gen_enc,gen_dec,enc

def evaluate():
    gen_enc,gen_dec,enc = get_models()
    _,ds = get_tf_dataset(bs=1)
    labels = []
    scores = []
    for x,y in tqdm(ds,desc="computing scores"):
        z = gen_enc(x)
        x_hat = gen_dec(z)
        z_hat = enc(x_hat)
        labels.append(np.argmax(y))
        scores.append(float(l_enc(z,z_hat)))

    max_score = max(scores)
    min_score = min(scores)
    scores = list(map(lambda x:(x+min_score)/(max_score-min_score),scores))

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in tqdm(range(len(scores)),desc="computing metrics"):
        if labels[i]==1 and scores[i]>TRESHOLD:
            tp +=1
        if labels[i]==0 and scores[i]<TRESHOLD:
            tn +=1
        if labels[i]==1 and scores[i]<=TRESHOLD:
            fn+=1
        if labels[i]==0 and scores[i]>=TRESHOLD:
            fp +=1

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall=tp/(tp+fn)
    print("#################################")
    print("accuracy:{}\nprecision:{}\nrecall:{}".format(accuracy,precision,recall))
    print("#################################")


if __name__=="__main__":
    evaluate()