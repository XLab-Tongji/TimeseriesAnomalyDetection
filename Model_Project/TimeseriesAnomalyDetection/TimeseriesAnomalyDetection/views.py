from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json


def index(request):
    return render(request, "index.html", {})


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

    print(algorithm + " / " + dataset)

    data_set = []
    original_anomaly = []
    detected_anomaly = []

    # 设置传给前端的相关数据
    data_set.append([2, 5])
    data_set.append([3, 6])
    data_set.append([4, 2])
    data_set.append([5, 7])
    data_set.append([7, 18])
    data_set.append([8, 45])
    data_set.append([10, 12])
    data_set.append([12, 14])
    data_set.append([14, 16])
    data_set.append([15, 10])
    data_set.append([16, 6])
    data_set.append([17, 8])
    data_set.append([18, 4])
    data_set.append([19, 12])
    data_set.append([20, 18])
    data_set.append([21, 26])
    data_set.append([22, 26])
    data_set.append([23, 16])
    data_set.append([24, 25])
    data_set.append([25, 32])
    data_set.append([28, 36])
    data_set.append([29, 24])
    data_set.append([30, 22])
    data_set.append([34, 12])
    data_set.append([35, 16])

    original_anomaly.append(7)
    original_anomaly.append(8)
    original_anomaly.append(15)
    original_anomaly.append(16)
    original_anomaly.append(17)
    original_anomaly.append(28)
    original_anomaly.append(29)

    detected_anomaly.append(15)
    detected_anomaly.append(16)
    detected_anomaly.append(17)
    detected_anomaly.append(22)
    detected_anomaly.append(23)
    detected_anomaly.append(29)

    retval = {
        "data_set": data_set,
        "original_anomaly": original_anomaly,
        "detected_anomaly": detected_anomaly
    }

    return HttpResponse(json.dumps(retval))
