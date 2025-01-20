import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import fastapi
from fastapi import HTTPException
import asyncio
from typing import List
import numpy as np
import torch.cuda.amp
import argparse
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelInstance:
    model: torch.nn.Module
    stream: torch.cuda.Stream
    in_use: bool = False

def parse_args():
    parser = argparse.ArgumentParser(description='FastAPI 기반 GPU 병렬 추론 서버')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--num-workers', type=int, default=16, help='워커(모델) 수 (기본값: 16)')
    return parser.parse_args()

app = fastapi.FastAPI(title="EfficientNet-B0 Inference API",
                     description="VOLTA GPU 최적화된 병렬 처리 이미지 분류 API")

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    model.to(device)
    
    # VOLTA의 Tensor Core 활용을 위한 설정
    if torch.cuda.get_device_capability()[0] >= 7:  # Volta 이상
        model = model.half()  # FP16 변환
    
    return model

# 모델 인스턴스들 초기화
model_instances: List[ModelInstance] = []

# ImageNet 클래스 레이블 로딩
import os
if not os.path.exists('imagenet_classes.txt'):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    with open('imagenet_classes.txt', 'w') as f:
        f.write(response.text)
with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

async def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지를 불러올 수 없습니다: {str(e)}")

async def get_available_model() -> Optional[ModelInstance]:
    """사용 가능한 모델 인스턴스를 찾아 반환합니다. 없으면 대기합니다."""
    while True:
        for instance in model_instances:
            if not instance.in_use:
                instance.in_use = True
                return instance
        # 사용 가능한 모델이 없으면 비동기 대기
        await asyncio.sleep(0.1)

async def run_inference(image: Image.Image, model_instance: ModelInstance):
    with torch.cuda.stream(model_instance.stream):
        try:
            # 이미지 전처리
            img_tensor = transform(image).unsqueeze(0).to(device)
            if torch.cuda.get_device_capability()[0] >= 7:
                img_tensor = img_tensor.half()
            
            # 추론
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model_instance.model(img_tensor)
            
            # 결과 처리
            probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            results = []
            for i in range(top5_prob.size(0)):
                results.append({
                    "category": categories[top5_catid[i]],
                    "probability": float(top5_prob[i])
                })
            
            return results
            
        finally:
            model_instance.in_use = False
            torch.cuda.empty_cache()

@app.get("/predict")
async def predict(image_url: str):
    """
    이미지 URL을 받아 EfficientNet-B0 모델로 추론을 수행합니다.
    여러 모델 인스턴스를 사용하여 병렬 처리를 수행합니다.
    
    Args:
        image_url (str): 추론할 이미지의 URL
    
    Returns:
        dict: 상위 5개 예측 결과와 확률
    """
    # 이미지 로딩
    image = await load_image_from_url(image_url)
    
    # 사용 가능한 모델 인스턴스 할당
    model_instance = await get_available_model()
    
    # 추론 실행
    results = await run_inference(image, model_instance)
    
    return {
        "predictions": results,
        "worker_id": model_instances.index(model_instance),
        "device_info": {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
            "memory_cached": f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
        }
    }

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업을 수행합니다."""
    global model_instances
    
    # CUDA 메모리 초기화
    torch.cuda.empty_cache()
    
    # 모델 인스턴스들 초기화
    for _ in range(NUM_WORKERS):
        print(f"Loading model {_}...")
        model = load_model()
        stream = torch.cuda.Stream()
        model_instances.append(ModelInstance(model=model, stream=stream))

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port) 