from TimeseriesAnomalyDetection.TSI_CNN.dataset.tf_dataset import get_tf_dataset
from TimeseriesAnomalyDetection.TSI_CNN.model.models import get_cnn as get_model
import tensorflow as tf
import os

model = get_model()
model.summary()
train_ds, val_ds = get_tf_dataset()

# compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy", tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
)

# early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
cb = [es]

# train
model.fit(x=train_ds, validation_data=val_ds, validation_steps=None, epochs=100, callbacks=cb)

# save trained model
path = os.path.join("tmp", "supervised")
if not os.path.exists(path):
    os.makedirs(path)
model.save(path)
