from TimeseriesAnomalyDetection.ECG.dataset.tf_dataset import get_tf_dataset
from TimeseriesAnomalyDetection.ECG.model.models import get_cnn
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


def evaluate(eval_index=0):
    _, ds = get_tf_dataset()
    path = "/workspace/AnomalyDetection/TimeseriesAnomalyDetection/ECG/tmp/supervised/saved_model.pb"
    model = tf.keras.models.load_model(path)

    total = 0
    success = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    count = 0
    msg = ""

    for x, y in tqdm(ds, desc="evaluating"):

        if count != eval_index:
            count += 1
            continue
        count += 1

        y_pred = model(x)
        for i in range(y.shape[0]):
            total += 1
            print("y_true:{},y_pred:{}".format(y[i], y_pred[i]))
            l_pred = np.argmax(y[i])
            l_true = np.argmax(y_pred[i])
            if l_pred == l_true:
                success += 1
                if l_pred == 0:
                    msg = "假阳性"
                    tp += 1
                else:
                    # 该数据为异常，且成功检测
                    msg = "真阳性"
                    tn += 1
            else:
                if l_pred == 0:
                    msg = "假阴性"
                    # 该数据为异常，且未检测出
                    fp += 1
                else:
                    msg = "真阴性"
                    fn += 1

            if count == eval_index + 1:
                break

    print("#################################")
    accuracy = success / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("accuracy:{}\nprecision:{}\nrecall:{}".format(accuracy, precision, recall))
    print("#################################")

    return msg


evaluate()

