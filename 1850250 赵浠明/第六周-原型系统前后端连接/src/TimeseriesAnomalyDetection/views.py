from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from TimeseriesAnomalyDetection.ECG.dataset.tf_dataset import send_dataset, send_dataset_param
from TimeseriesAnomalyDetection.ECG.model.evaluate import evaluate
from TimeseriesAnomalyDetection.RNN.readData import getRnn
import json


def index(request):
    return render(request, "index.html", {})


@csrf_exempt
def get_windows(request):
    post_content = request.POST
    model_sel = post_content.get("model")
    print(model_sel)

    if model_sel == 'CNN':
        data_size, window_size, max_selected = send_dataset_param()
    elif model_sel == 'RNN':
        data_size = 0
        window_size = 0
        max_selected = 1

    retval = {
        "data_size": data_size,
        "window_size": window_size,
        "max_selected": max_selected
    }
    return HttpResponse(json.dumps(retval))


@csrf_exempt
def get_data(request):
    # AJAX with vue
    # post_content = json.loads(request.body)

    # AJAX with jQuery
    # post_content = request.POST

    # 后端取到前端发来的信息（模型算法与数据集），此处使用jQuery的Ajax方法完成交互
    post_content = request.POST
    algorithm = post_content.get("algorithm")
    # dataset = post_content.get("dataset")
    windows = post_content.getlist("windows")

    # print(algorithm + " / " + dataset)
    print(algorithm)
    print(windows)

    data_set = []
    original_anomaly = []
    detected_anomaly = []
    anomaly_score = []
    res_value = ""

    # 设置传给前端的相关数据
    if algorithm == 'CNN':
        data_set = send_dataset(int(windows[0]))
        res_value = evaluate(int(windows[0]))
    elif algorithm == 'RNN':
        data_set, original_anomaly, anomaly_score = getRnn()

    retval = {
        "data_set": data_set,
        "original_anomaly": original_anomaly,
        # "detected_anomaly": detected_anomaly,
        "anomaly_score": anomaly_score,
        "res_value": res_value
    }

    return HttpResponse(json.dumps(retval))
