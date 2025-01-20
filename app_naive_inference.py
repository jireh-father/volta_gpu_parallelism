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
import atexit
import signal
import time

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
request_queues = []
processes = []

def initialize_workers():
    """워커 프로세스들을 초기화합니다."""
    global processes, request_queues
    
    # ImageNet 클래스 파일 다운로드 (없는 경우)
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        with open('imagenet_classes.txt', 'w') as f:
            f.write(response.text)
    
    # 워커 프로세스와 큐 초기화
    request_queues = [Queue() for _ in range(NUM_WORKERS)]
    
    # 워커 프로세스 시작
    for queue in request_queues:
        p = Process(target=worker_process, args=(queue,))
        p.daemon = True  # 메인 프로세스가 종료되면 자동으로 종료되도록 설정
        p.start()
        processes.append(p)

def cleanup_workers():
    """서버 종료 시 정리 작업을 수행합니다."""
    global processes, request_queues
    
    # 워커 프로세스들에게 종료 시그널 전송
    for queue in request_queues:
        try:
            if not queue._closed:
                queue.put(None)
        except (ValueError, AttributeError):
            continue
    
    # 모든 프로세스 종료 대기
    for p in processes:
        try:
            if p.is_alive():
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()
        except:
            continue

def signal_handler(signum, frame):
    """시그널 핸들러"""
    cleanup_workers()
    os._exit(0)

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    
    while True:
        try:
            # 가장 적은 대기열을 가진 큐 선택
            selected_queue = min(request_queues, key=lambda q: q.qsize())
            
            # 모든 큐가 너무 바쁜지 확인 (예: 큐 크기가 100 이상)
            if selected_queue.qsize() >= 100:
                time.sleep(0.1)  # 100ms 대기
                continue
            
            # 이미지 URL을 큐에 추가
            selected_queue.put(image_url)
            
            # 결과 대기
            result = selected_queue.get()
            
            # 결과가 문자열인 경우 (에러 메시지)
            if isinstance(result, str):
                print(f"Error: {result}", flush=True)
                return jsonify({"error": result}), 500
            
            # 결과가 딕셔너리인 경우 (정상 결과)
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
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 워커 초기화
    initialize_workers()
    
    # 종료 시 정리 작업 등록
    atexit.register(cleanup_workers)
    
    # 서버 실행
    app.run(host=args.host, port=args.port, threaded=True) 