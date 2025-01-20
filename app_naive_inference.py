import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from multiprocessing import Process, Queue, Value, current_process
import numpy as np
import os
import argparse
import atexit
import signal
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Flask 기반 멀티프로세스 추론 서버')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--num-workers', type=int, default=16, help='워커(모델) 수 (기본값: 16)')
    return parser.parse_args()

# 명령행 인자 파싱
args = parse_args()
NUM_WORKERS = args.num_workers

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def load_model():
    """모델을 로드하고 최적화합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    model.to(device)
    
    # FP16 변환 (VOLTA 이상)
    if torch.cuda.get_device_capability()[0] >= 7:
        model = model.half()
    
    return model, device

def worker_process(worker_id: int, request_queue: Queue, response_queue: Queue):
    """각 워커 프로세스에서 실행되는 함수"""
    print(f"Worker {worker_id} initializing...")
    
    # 모델과 디바이스 초기화
    model, device = load_model()
    
    # ImageNet 클래스 로드
    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    
    print(f"Worker {worker_id} ready!")
    
    while True:
        try:
            # 요청 대기
            request = request_queue.get()
            if request is None:  # 종료 신호
                break
                
            image_url = request["url"]
            task_id = request["task_id"]
            
            try:
                # 이미지 다운로드 및 전처리
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # FP16 변환 (VOLTA 이상)
                if torch.cuda.get_device_capability()[0] >= 7:
                    img_tensor = img_tensor.half()
                
                # 추론
                with torch.no_grad(), torch.cuda.amp.autocast():
                    output = model(img_tensor)
                
                # 결과 처리
                probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                
                results = []
                for i in range(top5_prob.size(0)):
                    results.append({
                        "category": categories[top5_catid[i]],
                        "probability": float(top5_prob[i])
                    })
                
                response_queue.put({
                    "task_id": task_id,
                    "success": True,
                    "worker_id": worker_id,
                    "predictions": results,
                    "device_info": {
                        "name": torch.cuda.get_device_name(),
                        "capability": torch.cuda.get_device_capability(),
                        "memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
                        "memory_cached": f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
                    }
                })
                
            except Exception as e:
                response_queue.put({
                    "task_id": task_id,
                    "success": False,
                    "worker_id": worker_id,
                    "error": str(e)
                })
            
            finally:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}")
            continue

# 프로세스와 큐 초기화
processes = []
request_queues = []
response_queue = Queue()

app = Flask(__name__)

@app.route('/')
def home():
    """루트 경로 처리"""
    return jsonify({
        "name": "EfficientNet-B0 Inference Server",
        "endpoints": {
            "/predict": "GET request with image_url parameter for inference",
            "/health": "GET request for health check"
        }
    })

@app.route('/predict', methods=['GET'])
def predict():
    """이미지 URL을 받아 추론을 수행합니다."""
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "이미지 URL이 필요합니다."}), 400
    
    # 가장 적은 대기열을 가진 큐 선택
    selected_queue = min(request_queues, key=lambda q: q.qsize())
    task_id = f"{time.time()}-{os.getpid()}"
    
    # 요청 전송
    selected_queue.put({"url": image_url, "task_id": task_id})
    
    # 응답 대기
    while True:
        response = response_queue.get()
        if response["task_id"] == task_id:
            if response["success"]:
                return jsonify({
                    "predictions": response["predictions"],
                    "worker_id": response["worker_id"],
                    "device_info": response["device_info"]
                })
            else:
                return jsonify({"error": response["error"]}), 500
        else:
            # 다른 태스크의 응답이면 다시 큐에 넣기
            response_queue.put(response)
            time.sleep(0.1)

def cleanup():
    """서버 종료 시 정리 작업을 수행합니다."""
    print("Cleaning up...")
    # 워커 프로세스들에게 종료 신호 전송
    for q in request_queues:
        q.put(None)
    
    # 프로세스 종료 대기
    for p in processes:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
    
    print("Cleanup complete!")

def initialize():
    """서버 시작 시 초기화 작업을 수행합니다."""
    global processes, request_queues
    
    # ImageNet 클래스 파일 다운로드 (없는 경우)
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        with open('imagenet_classes.txt', 'w') as f:
            f.write(response.text)
    
    # 워커 프로세스 시작
    for i in range(NUM_WORKERS):
        request_queue = Queue()
        request_queues.append(request_queue)
        p = Process(target=worker_process, args=(i, request_queue, response_queue))
        p.start()
        processes.append(p)
    
    # 종료 시 정리 작업 등록
    atexit.register(cleanup)

if __name__ == '__main__':
    # 초기화 실행
    initialize()
    
    # 서버 실행
    app.run(host=args.host, port=args.port, threaded=True) 