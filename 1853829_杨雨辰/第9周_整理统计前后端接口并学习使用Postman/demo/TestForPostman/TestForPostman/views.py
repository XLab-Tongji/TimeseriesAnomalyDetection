from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json


def index(request):
    return render(request, "index.html", {})


@csrf_exempt
def get_data(request):
    post_ori = request.POST
    ori_text = post_ori.get("text")
    ori_price = int(post_ori.get("price"))
    arr = []
    for i in range(ori_price):
        arr.append([i, i + ori_price])
    retval = {
        'text': 'text: ' + ori_text,
        'price': 'price: ' + str(ori_price),
        'array': arr
    }
    return HttpResponse(json.dumps(retval))
