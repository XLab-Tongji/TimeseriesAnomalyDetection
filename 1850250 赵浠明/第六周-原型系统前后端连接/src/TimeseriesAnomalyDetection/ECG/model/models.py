import tensorflow as tf

SEQ_LEN = 188

def get_cnn():
    input = tf.keras.layers.Input(shape=(SEQ_LEN,SEQ_LEN,3))
    x = tf.keras.layers.Conv2D(64,(5,5),activation='relu')(input)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128,(5,5),activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(256,(5,5),activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(512,(5,5),activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2,activation="softmax")(x)
    return tf.keras.Model(inputs=input,outputs=x)

if __name__=="__main__":
    m = get_cnn()
    m.summary()
    pass