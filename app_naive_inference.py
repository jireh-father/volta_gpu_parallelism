import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from multiprocessing import Process, Value, cpu_count
import numpy as np
import os
import argparse
import atexit
import signal
import time
import ctypes

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

class InferenceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.busy = Value(ctypes.c_bool, False)
        
        # ImageNet 클래스 로드
        with open('imagenet_classes.txt', 'r') as f:
            self.categories = [s.strip() for s in f.readlines()]
    
    def _load_model(self):
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.eval()
        model.to(self.device)
        
        # FP16 변환 (VOLTA 이상)
        if torch.cuda.get_device_capability()[0] >= 7:
            model = model.half()
        
        return model
    
    def infer(self, image_url: str):
        try:
            # 이미지 다운로드 및 전처리
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # FP16 변환 (VOLTA 이상)
            if torch.cuda.get_device_capability()[0] >= 7:
                img_tensor = img_tensor.half()
            
            # 추론
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = self.model(img_tensor)
            
            # 결과 처리
            probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            results = []
            for i in range(top5_prob.size(0)):
                results.append({
                    "category": self.categories[top5_catid[i]],
                    "probability": float(top5_prob[i])
                })
            
            return {"success": True, "predictions": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            torch.cuda.empty_cache()

# 모델 인스턴스들을 저장할 리스트
models = []

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
    
    while True:
        # 사용 가능한 모델 찾기
        for model in models:
            if not model.busy.value:
                with model.busy.get_lock():
                    if not model.busy.value:  # Double-check
                        model.busy.value = True
                        try:
                            result = model.infer(image_url)
                            if result["success"]:
                                return jsonify({
                                    "predictions": result["predictions"],
                                    "worker_id": models.index(model),
                                    "device_info": {
                                        "name": torch.cuda.get_device_name(),
                                        "capability": torch.cuda.get_device_capability(),
                                        "memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
                                        "memory_cached": f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
                                    }
                                })
                            else:
                                return jsonify({"error": result["error"]}), 500
                        finally:
                            model.busy.value = False
        
        # 사용 가능한 모델이 없으면 잠시 대기
        time.sleep(0.1)

@app.before_first_request
def initialize():
    """서버 시작 시 초기화 작업을 수행합니다."""
    global models
    
    # ImageNet 클래스 파일 다운로드 (없는 경우)
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        with open('imagenet_classes.txt', 'w') as f:
            f.write(response.text)
    
    # 모델 인스턴스들 초기화
    for _ in range(NUM_WORKERS):
        models.append(InferenceModel())

if __name__ == '__main__':
    # 서버 실행
    app.run(host=args.host, port=args.port, threaded=True) 