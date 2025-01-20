# GPU 병렬 처리를 활용한 EfficientNet-B0 추론 서버

이 프로젝트는 CUDA 스트림을 활용하여 GPU 병렬 처리 기능을 구현한 이미지 분류 추론 서버입니다. EfficientNet-B0 모델을 사용하여 이미지 분류를 수행합니다.

## 주요 기능

- EfficientNet-B0 pretrained 모델을 사용한 이미지 분류
- CUDA 스트림을 활용한 GPU 병렬 처리
- FastAPI 기반의 RESTful API
- 이미지 URL을 통한 간편한 추론

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU
- CUDA Toolkit 11.0 이상

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd [repository-name]
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 서버 실행
```bash
python app.py
```

2. API 문서 확인
- http://localhost:8000/docs 접속

## API 엔드포인트

### GET /predict

이미지 URL을 받아 분류 결과를 반환합니다.

**Parameters:**
- `image_url` (string): 분류할 이미지의 URL

**Response:**
```json
{
    "predictions": [
        {
            "category": "카테고리명",
            "probability": 0.123
        },
        ...
    ],
    "stream_idx": 0
}
```

## 성능 최적화

- 8개의 CUDA 스트림을 사용하여 병렬 처리
- 비동기 처리를 통한 효율적인 리소스 관리
- 스트림 상태 추적을 통한 작업 분배

## 라이선스

MIT License 