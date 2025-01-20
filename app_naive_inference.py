import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Flask 기반 멀티프로세스 추론 서버')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--num-workers', type=int, default=16, help='워커 프로세스 수 (기본값: 16)')
    return parser.parse_args()

# 명령행 인자 파싱
args = parse_args()
NUM_WORKERS = args.num_workers  # 사용자 지정 워커 수 사용

# 워커 프로세스와 큐 초기화
request_queues = [Queue() for _ in range(NUM_WORKERS)]
response_queue = Queue()
processes = []

def initialize_workers():
    """워커 프로세스들을 초기화합니다."""
    global processes
    
    # ImageNet 클래스 파일 다운로드 (없는 경우)
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        with open('imagenet_classes.txt', 'w') as f:
            f.write(response.text)
    
    # 워커 프로세스 시작
    for queue in request_queues:
        p = Process(target=worker_process, args=(queue,))
        p.start()
        processes.append(p)

app = Flask(__name__)

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def load_model():
    """각 프로세스에서 독립적으로 모델을 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    model.to(device)
    
    # FP16 변환 (VOLTA 이상)
    if torch.cuda.get_device_capability()[0] >= 7:
        model = model.half()
    
    return model, device

def worker_process(queue):
    """워커 프로세스에서 실행되는 추론 함수"""
    # 모델과 디바이스 초기화
    model, device = load_model()
    
    # ImageNet 클래스 로드
    with open('imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    
    while True:
        # 큐에서 이미지 URL 가져오기
        image_url = queue.get()
        if image_url is None:  # 종료 시그널
            break
            
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
            
            queue.put({"success": True, "predictions": results})
            
        except Exception as e:
            queue.put({"success": False, "error": str(e)})
        
        # 메모리 정리
        torch.cuda.empty_cache()

@app.route('/predict', methods=['GET'])
def predict():
    """이미지 URL을 받아 추론을 수행합니다."""
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "이미지 URL이 필요합니다."}), 400
    
    # 가장 적은 대기열을 가진 큐 선택
    selected_queue = min(request_queues, key=lambda q: q.qsize())
    
    # 이미지 URL을 큐에 추가
    selected_queue.put(image_url)
    
    # 결과 대기
    result = selected_queue.get()
    
    if result.get("success", False):
        return jsonify({
            "predictions": result["predictions"],
            "device_info": {
                "name": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
                "memory_cached": f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
            }
        })
    else:
        return jsonify({"error": result.get("error", "알 수 없는 오류가 발생했습니다.")}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    """서버 종료 시 정리 작업을 수행합니다."""
    # 워커 프로세스들에게 종료 시그널 전송
    for queue in request_queues:
        queue.put(None)
    
    # 모든 프로세스 종료 대기
    for p in processes:
        p.join()
    
    # 큐 정리
    for queue in request_queues:
        queue.close()
    response_queue.close()

if __name__ == '__main__':
    # 워커 초기화
    initialize_workers()
    
    # 서버 실행
    app.run(host=args.host, port=args.port, threaded=True) 