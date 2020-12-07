import model.unsupervised.models as models
from dataset.tf_dataset import plot_sample,get_tf_dataset
import os
import tensorflow as tf
import numpy as np

def inspect():
    base_path = os.path.join("tmp","unsupervised","models")
    if not os.path.exists(base_path):
        raise IOError("Path {} does not exists".format(base_path))
    
    gen_dec = tf.keras.models.load_model(os.path.join(base_path,"gen_dec"))
    gen_enc = tf.keras.models.load_model(os.path.join(base_path,"gen_enc"))
    enc = tf.keras.models.load_model(os.path.join(base_path,"enc"))

    ds,_ = get_tf_dataset(bs=1)
    for x,y in ds:
        print("label:{}".format(np.argmax(y[0])))
        z = gen_enc(x)
        d = gen_dec(z)
        e = enc(x)
        d_rec = gen_dec(e)

        x = x[0].numpy()
        d = d[0].numpy()
        d_rec = d_rec[0].numpy()
        print("real")
        plot_sample(x)
        print("generated")
        plot_sample(d)
        print("reconstructed from encoder")
        plot_sample(d_rec)


if __name__=="__main__":
    inspect()