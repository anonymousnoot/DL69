from django.shortcuts import render
import json
import queue
import threading
from django.http import StreamingHttpResponse
from .ml.train import train

# Create your views here.

def index(request):
    return render(request, 'dashboard/index.html')

def train_stream(request):
    def event_stream():
        q = queue.Queue()

        def progress_callback(epoch, loss, acc):
            q.put({"type": "progress", "epoch": epoch, "loss": loss, "accuracy": acc})

        def run_training():
            summary = train(on_progress=progress_callback)
            q.put({"type": "done", "summary": summary})

        thread = threading.Thread(target=run_training)
        thread.start()

        while True:
            item = q.get()
            data = json.dumps(item)
            yield f"data: {data}\n\n"
            if item.get("type") == "done":
                break

        thread.join()

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response