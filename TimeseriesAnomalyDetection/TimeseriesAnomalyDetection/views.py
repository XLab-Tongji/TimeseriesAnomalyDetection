import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from . import settings
from TimeseriesAnomalyDetection.TSI_CNN.dataset.tf_dataset_eval import init_val, load_file, send_dataset, send_dataset_param
from TimeseriesAnomalyDetection.TSI_CNN.model.evaluate import evaluate
from TimeseriesAnomalyDetection.RNN.readData import get_RNN_result
import json


def index(request):
    return render(request, "index.html", {})


@csrf_exempt
def get_windows(request):
    post_content = request.POST
    file_name = post_content.get("file")
    print(post_content)
    print(file_name)
    if len(file_name) == 0:
        init_val()
    else:
        load_file(file_name)
    data_size, window_size = send_dataset_param()
    retval = {
        "data_size": data_size,
        "window_size": window_size,
    }
    return HttpResponse(json.dumps(retval))


@csrf_exempt
def get_files(request):
    retval = ["original data"]
    file_dir = "./TimeseriesAnomalyDetection/TSI_CNN/dataset/data/upload_data/"
    for root, dirs, files in os.walk(file_dir):
        for item in files:
            if item.endswith(".csv"):
                retval.append(item)
    return HttpResponse(json.dumps(retval))


@csrf_exempt
def save_file(request):
    file = request.FILES.get("file")
    if file:
        dir = os.path.join(settings.BASE_DIR, "TimeseriesAnomalyDetection\\TSI_CNN\\dataset\\data\\upload_data\\")
        print(dir)
        destination = open(os.path.join(dir, file.name), 'wb+')
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        return HttpResponse(json.dumps("success"))
    else:
        return HttpResponse(json.dumps("error"))


@csrf_exempt
def get_data(request):
    # AJAX with vue
    # post_content = json.loads(request.body)

    # AJAX with jQuery
    # post_content = request.POST

    # 后端取到前端发来的信息（模型算法与数据集），此处使用jQuery的Ajax方法完成交互
    post_content = request.POST
    algorithm = post_content.get("algorithm")
    dataset = post_content.get("dataset")
    window = post_content.get("window")
    model_type = post_content.get("model_type")
    file_name = post_content.get("file")

    print(post_content)

    data_set = []
    original_anomaly = []
    detect_res = []
    original_value = "0"
    res_value = "0"
    auc_val = ""

    # 设置传给前端的相关数据
    if model_type == 'true':
        data_set = send_dataset(int(window))
        original_value, res_value = evaluate(int(window), file_name)
    else:
        data_set, original_anomaly, detect_res, auc_val = get_RNN_result(dataset.lower(), algorithm)

    retval = {
        "data_set": data_set,
        "original_anomaly": original_anomaly,
        "detect_res": detect_res,
        "original_value": original_value,
        "res_value": res_value,
        "auc": auc_val
    }

    print(retval)

    return HttpResponse(json.dumps(retval))
