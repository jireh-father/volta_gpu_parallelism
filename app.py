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
import torch.cuda.amp  # Mixed Precision 추가
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FastAPI 기반 GPU 병렬 추론 서버')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--num-streams', type=int, default=16, help='CUDA 스트림 수 (기본값: 16)')
    return parser.parse_args()

app = fastapi.FastAPI(title="EfficientNet-B0 Inference API",
                     description="VOLTA GPU 최적화된 병렬 스트림 이미지 분류 API")

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 명령행 인자 파싱
args = parse_args()
MAX_STREAMS = args.num_streams  # 사용자 지정 스트림 수 사용

# Mixed Precision 설정
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 모델 로딩 및 VOLTA 최적화
model = torchvision.models.efficientnet_b0(pretrained=True)
model.eval()
model.to(device)

# VOLTA의 Tensor Core 활용을 위한 설정
if torch.cuda.get_device_capability()[0] >= 7:  # Volta 이상
    model = model.half()  # FP16 변환

# ImageNet 클래스 레이블 로딩
with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

# GPU 스트림 설정
streams = [torch.cuda.Stream() for _ in range(MAX_STREAMS)]
stream_status = [False] * MAX_STREAMS

# 메모리 풀 초기화
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()

async def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지를 불러올 수 없습니다: {str(e)}")

def get_available_stream():
    for i, status in enumerate(stream_status):
        if not status:
            stream_status[i] = True
            return i
    return -1

async def run_inference_in_stream(image: Image.Image, stream_idx: int):
    with torch.cuda.stream(streams[stream_idx]):
        try:
            # 이미지 전처리
            img_tensor = transform(image).unsqueeze(0).to(device)
            if torch.cuda.get_device_capability()[0] >= 7:
                img_tensor = img_tensor.half()  # VOLTA 이상에서는 FP16 사용
            
            # 추론 (Mixed Precision 사용)
            with torch.no_grad(), autocast():
                output = model(img_tensor)
            
            # 결과 처리
            probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)  # float()로 변환하여 안정성 확보
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            results = []
            for i in range(top5_prob.size(0)):
                results.append({
                    "category": categories[top5_catid[i]],
                    "probability": float(top5_prob[i])
                })
            
            return results
            
        finally:
            stream_status[stream_idx] = False
            # 메모리 정리
            torch.cuda.empty_cache()

@app.get("/predict")
async def predict(image_url: str):
    """
    이미지 URL을 받아 EfficientNet-B0 모델로 추론을 수행합니다.
    VOLTA GPU에 최적화된 Mixed Precision과 병렬 스트림을 사용합니다.
    
    Args:
        image_url (str): 추론할 이미지의 URL
    
    Returns:
        dict: 상위 5개 예측 결과와 확률
    """
    # 스트림 할당
    stream_idx = get_available_stream()
    if stream_idx == -1:
        raise HTTPException(status_code=503, detail="사용 가능한 스트림이 없습니다. 잠시 후 다시 시도해주세요.")
    
    # 이미지 로딩
    image = await load_image_from_url(image_url)
    
    # 추론 실행
    results = await run_inference_in_stream(image, stream_idx)
    
    return {
        "predictions": results,
        "stream_idx": stream_idx,
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
    global stream_status
    stream_status = [False] * MAX_STREAMS
    
    # CUDA 메모리 초기화
    torch.cuda.empty_cache()
    
    # ImageNet 클래스 파일 다운로드 (없는 경우)
    import os
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        with open('imagenet_classes.txt', 'w') as f:
            f.write(response.text)

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port) 