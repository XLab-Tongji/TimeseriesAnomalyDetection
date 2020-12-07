'''
This module contains methods to build the tensorflow dataset and to
plot the dataset sample
-> get_tf_dataset: build the tensorflow dataset
-> plot_sample: plot a sample
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField,MarkovTransitionField

# the lenght of the ecg time serie
SEQ_LEN = 188

def shuffle_in_unison(a, b,seed=None):
    """
    shuffle arrays a and b with the same
    permutation of indexes, use seed to
    make the permutation reproducible
    """
    if seed!=None:
        np.random.seed(seed=seed)
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def gen(split="train",sample_type="mixed"):
    # load normal and abnormal egc from csv files
    normal = np.array(pd.read_csv("/Users/kerrzhang/Downloads/ECG-Anomaly-Detection-master/dataset/data/normal.csv"))
    abnormal = np.array(pd.read_csv("/Users/kerrzhang/Downloads/ECG-Anomaly-Detection-master/dataset/data/abnormal.csv"))
    # balance the dataset
    min_shape = min(normal.shape[0],abnormal.shape[0])-1
    normal = normal[0:min_shape]
    abnormal = abnormal[0:min_shape]
    # get the requested split (train or validation)
    if split=="train":
        normal = normal[0:int(normal.shape[0]*0.8)]
        abnormal = abnormal[0:int(abnormal.shape[0]*0.8)]
    elif split=="val":
        normal = normal[int(normal.shape[0]*0.8):]
        abnormal = abnormal[int(abnormal.shape[0]*0.8):]
    else:
        raise ValueError("split should be one of [train|val]")

    # generate labels
    normal_labels = np.zeros((normal.shape[0]))      
    abnormal_labels = np.ones((abnormal.shape[0]))

    if sample_type=="mixed":
        X = np.concatenate((normal,abnormal))
        Y = np.concatenate((normal_labels,abnormal_labels))
    elif sample_type=="normal":
        X = normal
        Y = normal_labels
    elif sample_type=="abnormal":
        X = abnormal
        Y = abnormal_labels
    else:
        raise ValueError("sample_type should be one of [mixed|normal|abnormal]")

    X,Y = shuffle_in_unison(X,Y,seed=9)



    # for each ecg generate the visual representation,
    # that consists in 3 channels:
    # -> gramian summation angular field
    # -> gramian difference angular field
    # -> markov transition field
    gasf = GramianAngularField(image_size=1.0, method='summation')
    gadf = GramianAngularField(image_size=1.0, method='difference')
    mtf = MarkovTransitionField(image_size=1.0,n_bins=8,strategy='uniform')

    norm = None
    # generate visual representation for 10 series
    # at a time (for performance)
    for i in range(0,X.shape[0]-1,10):
        x_gasf = gasf.fit_transform(X[i:i+10])
        x_gadf = gadf.fit_transform(X[i:i+10])
        x_mtf = mtf.fit_transform(X[i:i+10])

        x = np.stack([x_gasf,x_gadf,x_mtf],axis=-1)
        y = Y[i:i+10]

        for j in range(x.shape[0]):
            if y[j]==0:
                yield (x[j]+1)/2,np.array([1,0],dtype=np.float32)
            else:
                yield (x[j]+1)/2,np.array([0,1],dtype=np.float32)


def get_tf_dataset(bs=32,sample_type="mixed"):
    """
    return two datasets (train_set,val_set). Each element is a tuple x,y, s.t.
    ->  x is a visual representation of the time serie. It's a 3 channel image where:
            - 1st channel: gramian summation angular field normalized in [0,1]
            - 2nd chennel: gramian difference angular field normalized in [0,1]
            - 3rd chennel: markov transition field normalized in [0,1]
        each channel has the same width,height that are both equal to the 
        length of the sequence. For example if the length of the series is N, then
        x.shape = (N,N,3)
    ->  y is a one hot encoding of the category, y.shape=(2,)
    """
    train_ds = tf.data.Dataset.from_generator(
        lambda :gen(split="train",sample_type=sample_type),
        (tf.float32,tf.float32),
        ((SEQ_LEN,SEQ_LEN,3),(2)))
    train_ds = train_ds.batch(bs,drop_remainder=True)

    val_ds = tf.data.Dataset.from_generator(
        lambda :gen(split="val",sample_type=sample_type),
        (tf.float32,tf.float32),
        ((SEQ_LEN,SEQ_LEN,3),(2)))
    val_ds = val_ds.batch(bs,drop_remainder=True)
    return train_ds,val_ds


def plot_sample(x):
    """
    plot each component of a gasf-gadf-mtf sample separately. x must
    be a numpy array with shape (SEQ_LEN,SEQ_LEN,3)
    """
    plt.imshow(x[:,:,0])
    plt.title("gasf")
    plt.colorbar()
    plt.show()

    plt.imshow(x[:,:,1])
    plt.title("gadf")
    plt.colorbar()
    plt.show()

    plt.imshow(x[:,:,2])
    plt.title("mtf")
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    tds = gen(split="eval",sample_type="mixed")
    for x,y in tds:
        print(y)
        plot_sample(x)