import tensorflow as tf
from dataset.tf_dataset import get_tf_dataset
import os
from model.unsupervised.models import *
import matplotlib.pyplot as plt

LAMBDA_ADV = 1.#0.6
LAMBDA_CON = 10**-4

def safe_log(x):
    return tf.math.log(x+10**-8)

def l_enc(z,z_hat):
    z = tf.keras.backend.flatten(z)
    z_hat = tf.keras.backend.flatten(z_hat)
    l1 = tf.norm(z-z_hat,ord=1)
    l1 = tf.math.reduce_mean(l1)
    return l1

def l_con(x,x_hat):
    x = tf.keras.backend.flatten(x)
    x_hat = tf.keras.backend.flatten(x_hat)
    l1 = tf.norm(x-x_hat,ord=1)
    l1 = tf.math.reduce_mean(l1)
    return l1

def l_adv(dis_fake_output):
    l =  safe_log(tf.ones_like(dis_fake_output) - dis_fake_output)
    return tf.math.reduce_mean(l)

def l_dis(dis_real_output,dis_fake_output):
    l =  -1.  * safe_log(dis_real_output) -1.  * safe_log(tf.ones_like(dis_fake_output) - dis_fake_output)
    return tf.math.reduce_mean(l)

def plot_loss(l,name,path):
    plt.suptitle(name)
    plt.xlabel("ep")
    plt.ylabel("loss")
    plt.plot(l)
    plt.savefig(path)
    plt.clf()

def train(epochs=10):

    ckpt = tf.train.Checkpoint(
        ep = tf.Variable(0),
        step = tf.Variable(0),
        gen_encoder = encoder(name="gen_enc"),
        gen_decoder = decoder(name="gen_dec"),
        encoder = encoder(),
        disc = discriminator(),
        gen_opt = tf.keras.optimizers.Adam(),
        encoder_opt = tf.keras.optimizers.Adam(),
        disc_opt = tf.keras.optimizers.Adam()
    )

    manager = tf.train.CheckpointManager(ckpt, os.path.join("tmp","unsupervised","checkpoints"), max_to_keep=2)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    gen_avg_loss = []
    enc_avg_loss = []
    dis_avg_loss = []

    gen_cur_loss = 0.
    enc_cur_loss = 0.
    dis_cur_loss =0.

    ds = iter(get_tf_dataset(sample_type="normal")[0])

    while int(ckpt.ep)<epochs:
        # training loop, try to get the next element
        # of the iterator, if the iterator is exausted then
        # increment the epoch number and reinitialize the iterator
        try:
            ckpt.step.assign_add(1)
            x,_ = next(ds)

            # training step
            with tf.GradientTape(persistent=True) as tape:
                z = ckpt.gen_encoder(x)
                x_hat = ckpt.gen_decoder(z)
                z_hat = ckpt.encoder(x_hat)
                dis_fake_output = ckpt.disc(x_hat)
                dis_real_output = ckpt.disc(x)

                # compute losses
                gen_loss = LAMBDA_CON*l_con(x,x_hat) + LAMBDA_ADV*l_adv(dis_fake_output)
                enc_loss = l_enc(z,z_hat)
                dis_loss = l_dis(dis_real_output,dis_fake_output)

            # compute gradients 
            gen_grad = tape.gradient(
                gen_loss,
                ckpt.gen_encoder.trainable_variables+ckpt.gen_decoder.trainable_variables
                )
            enc_grad = tape.gradient(enc_loss,ckpt.encoder.trainable_variables)
            dis_grad = tape.gradient(dis_loss,ckpt.disc.trainable_variables)

            # apply gradients
            ckpt.gen_opt.apply_gradients(
                zip(gen_grad,ckpt.gen_encoder.trainable_variables+ckpt.gen_decoder.trainable_variables)
            )
            ckpt.encoder_opt.apply_gradients(
                zip(enc_grad,ckpt.encoder.trainable_variables)
            )
            ckpt.disc_opt.apply_gradients(
                zip(dis_grad,ckpt.disc.trainable_variables)
            )

            # compute the incremental average of the current epoch
            gen_cur_loss = ((int(ckpt.step)-1)*gen_cur_loss + gen_loss)/int(ckpt.step)
            enc_cur_loss = ((int(ckpt.step)-1)*enc_cur_loss + enc_loss)/int(ckpt.step)
            dis_cur_loss = ((int(ckpt.step)-1)*dis_cur_loss + dis_loss)/int(ckpt.step)

            print("ep:{}/{},it:{},gen_loss:{},enc_loss:{},dis_loss:{}".format(
                int(ckpt.ep),epochs,
                int(ckpt.step),
                gen_loss,
                enc_loss,
                dis_loss
            ))
            

        # when the iterator is exausted increment the epoch number,
        # reinitialize the iterator and save the checkpoint
        except StopIteration:
            # increment epoch number
            ckpt.ep.assign_add(1)
            # initialize step number
            ckpt.step = tf.Variable(0)
            # initialize iterator
            ds = iter(get_tf_dataset(sample_type="normal")[0])

            # add the average loss of the epoch to the
            # average losses and update the plot
            gen_avg_loss.append(float(gen_cur_loss))
            enc_avg_loss.append(float(enc_cur_loss))
            dis_avg_loss.append(float(dis_cur_loss))

            base_path = os.path.join("tmp","unsupervised","losses")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            plot_loss(gen_avg_loss,"generator loss",os.path.join(base_path,"gen.png"))
            plot_loss(enc_avg_loss,"encoder loss",os.path.join(base_path,"enc.png"))
            plot_loss(dis_avg_loss,"discriminator loss",os.path.join(base_path,"dis.png"))

            gen_cur_loss = enc_cur_loss = dis_cur_loss = 0

            # save the checkpoint and if both generators and encoders losses
            # are decreased then save the models
            manager.save()
            base_path = os.path.join("tmp","unsupervised","models")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            if len(enc_avg_loss)>2 and gen_avg_loss[-1]<=min(gen_avg_loss) and enc_avg_loss[-1]<=min(enc_avg_loss):
                print("save model")
                ckpt.gen_encoder.save(os.path.join(base_path,"gen_enc"))
                ckpt.gen_decoder.save(os.path.join(base_path,"gen_dec"))
                ckpt.encoder.save(os.path.join(base_path,"enc"))
            continue


if __name__=="__main__":
    train(epochs=1000)