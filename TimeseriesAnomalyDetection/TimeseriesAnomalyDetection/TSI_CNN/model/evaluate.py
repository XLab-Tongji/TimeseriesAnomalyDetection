from TimeseriesAnomalyDetection.TSI_CNN.dataset.tf_dataset_eval import get_tf_dataset
from TimeseriesAnomalyDetection.TSI_CNN.model.models import get_cnn
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


def evaluate(eval_index=0, filename=""):
    tf.enable_eager_execution()

    ds = get_tf_dataset(filename=filename)
    # path = "./TimeseriesAnomalyDetection/TSI_CNN/tmp/supervised/saved_model.pb"
    path = "./TimeseriesAnomalyDetection/TSI_CNN/tmp/supervised/saved_model.pb"
    model = tf.keras.models.load_model(path)

    # print("############")
    # print("length of evaluate dataset:" + str(len(ds)))
    # print("############")

    total = 0
    success = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    count = 0
    origin_msg = ""
    msg = ""

    for x, y in tqdm(ds, desc="evaluating"):

        if eval_index >= count + 32:
            count = count + 32
            continue

        y_pred = model(x)
        for i in range(y.shape[0]):
            total += 1
            print("y_true:{},y_pred:{}".format(y[i], y_pred[i]))
            l_true = np.argmax(y[i])
            l_pred = np.argmax(y_pred[i])

            if l_true == 0:
                # origin_msg = "异常"
                origin_msg = "2"
            else:
                # origin_msg = "正常"
                origin_msg = "1"

            print("l_true:{},l_pred:{}".format(l_true, l_pred))
            if l_pred == l_true:
                success += 1
                if l_pred == 0:
                    # 该数据为异常，检测结果为异常
                    # msg = "异常"
                    msg = "2"
                    tp += 1
                else:
                    # 该数据为正常，检测结果为正常
                    # msg = "正常"
                    msg = "1"
                    tn += 1
            else:
                if l_pred == 0:
                    # 该数据为正常，检测结果为异常
                    # msg = "异常"
                    msg = "2"
                    fp += 1
                else:
                    # 该数据为异常，检测结果为正常
                    # msg = "正常"
                    msg = "1"
                    fn += 1

            if count == eval_index:
                break
            count = count + 1

        if count == eval_index:
            break

    print(count)
    print("#################################")
    # accuracy = success / total
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # print("accuracy:{}\nprecision:{}\nrecall:{}".format(accuracy, precision, recall))
    print(msg)
    print("#################################")

    return origin_msg, msg

