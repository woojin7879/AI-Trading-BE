# AI-Trading-BE

야후 파이낸스 API를 이용한 주식 AI 자동 매매 시스템

## 프로젝트 개요

이 프로젝트는 야후 파이낸스 API를 이용하여 주식 데이터를 수집하고, 머신러닝/딥러닝 모델을 통해 주가 방향을 예측하여 자동으로 매매하는 시스템입니다. Alpaca API를 통해 실제 거래를 수행할 수 있으며, 페이퍼 트레이딩 모드도 지원합니다.

## 주요 기능

- 야후 파이낸스 API를 통한 주식 데이터 수집
- 기술적 지표 계산 및 데이터 전처리
- 머신러닝/딥러닝 모델을 이용한 주가 방향 예측
- 백테스팅을 통한 전략 성능 평가
- Alpaca API를 이용한 자동 매매 실행
- 페이퍼 트레이딩 및 실제 거래 지원

## 시스템 구조

```
AI-Trading-BE/
├── data/                  # 데이터 저장 디렉토리
│   ├── raw/               # 원시 데이터
│   ├── processed/         # 처리된 데이터
│   ├── backtest/          # 백테스트 결과
│   └── trades/            # 거래 기록
├── logs/                  # 로그 파일
├── models/                # 학습된 모델
├── src/                   # 소스 코드
│   ├── config/            # 설정 파일
│   ├── data/              # 데이터 관련 모듈
│   ├── models/            # 모델 관련 모듈
│   ├── strategies/        # 트레이딩 전략
│   ├── trading/           # 트레이딩 실행 모듈
│   ├── utils/             # 유틸리티 함수
│   └── backtesting/       # 백테스팅 모듈
├── .env                   # 환경 변수 (API 키 등)
├── .env.example           # 환경 변수 예제
├── requirements.txt       # 필요 패키지
└── main.py                # 메인 실행 파일
```

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/AI-Trading-BE.git
cd AI-Trading-BE
```

2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 필요한 API 키 등을 설정합니다.

```bash
cp .env.example .env
```

## 사용 방법

### 데이터 수집

```bash
python main.py collect --symbols AAPL MSFT GOOGL
```

### 데이터 처리

```bash
python main.py process --symbols AAPL MSFT GOOGL
```

### 모델 학습

```bash
python main.py train --symbols AAPL --model-type dense --epochs 100
```

### 백테스트

```bash
python main.py backtest --symbols AAPL --start-date 2022-01-01 --end-date 2023-01-01
```

### 트레이딩 시작

```bash
# 페이퍼 트레이딩
python main.py trade --symbols AAPL MSFT GOOGL --interval 60

# 실제 거래
python main.py trade --symbols AAPL MSFT GOOGL --live --interval 60
```

### 전체 파이프라인 실행

```bash
python main.py pipeline --symbols AAPL MSFT GOOGL --model-type dense
```

## 모델 유형

- `dense`: 기본 밀집 신경망 모델
- `lstm`: LSTM 기반 순환 신경망 모델
- `ensemble`: 밀집 신경망과 LSTM을 결합한 앙상블 모델

## 필요 API 키

- Alpaca API: 주식 거래를 위한 API 키 ([Alpaca 웹사이트](https://alpaca.markets/)에서 발급)

## 주의사항

- 이 시스템은 투자 조언을 제공하지 않으며, 실제 투자에 사용할 경우 발생하는 손실에 대해 책임지지 않습니다.
- 실제 거래 전에 반드시 페이퍼 트레이딩을 통해 충분히 테스트하세요.
- API 사용량 제한을 고려하여 데이터 수집 빈도를 조절하세요.

## 라이선스

MIT License
