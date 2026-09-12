import json
import os
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from .ml import train


def index(request):
    """Landing page – ใส่รหัสนักศึกษาและชื่อที่นี่"""
    context = {
        "student_id": "67114540282",
        "student_name": "นายนพวัชร พรสุข",
    }
    return render(request, "dashboard/index.html", context)


def train_stream(request):
    """SSE: สตรีมผลการฝึกสอนแบบ real-time รองรับ ?model=lstm หรือ ?model=rnn"""
    model_type = request.GET.get("model", "lstm")

    def event_stream():
        for event in train.event_stream(model_type=model_type):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")


def load_results(request):
    """หน้า/endpoint โหลดผลลัพธ์กราฟและสถานะล่าสุด"""
    static_dir = os.path.join("dashboard", "static", "dashboard")
    pred_path = os.path.join(static_dir, "prediction_plot.png")
    loss_path = os.path.join(static_dir, "loss_curve.png")

    return JsonResponse({
        "status": "ready" if os.path.exists(pred_path) else "empty",
        "prediction_plot": "/static/dashboard/prediction_plot.png" if os.path.exists(pred_path) else None,
        "loss_curve": "/static/dashboard/loss_curve.png" if os.path.exists(loss_path) else None,
    })