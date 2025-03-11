import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# 환경 변수 로드
load_dotenv()

# 기본 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# API 설정
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# 거래 설정
TRADING_MODE = os.getenv("TRADING_MODE", "paper")
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 5000))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 0.02))

# 모델 설정
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_PATH = os.getenv("MODEL_PATH", "models/")

# 데이터베이스 설정
DB_PATH = os.getenv("DB_PATH", "data/trading.db")

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/")

# 로깅 설정 함수
def setup_logging(name=None):
    log_level = getattr(logging, LOG_LEVEL)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 파일 경로
    log_file = LOG_DIR / f"{name or 'app'}.log"
    
    # 로거 설정
    logger = logging.getLogger(name or __name__)
    logger.setLevel(log_level)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 기본 로거 설정
logger = setup_logging()

# 거래할 기본 주식 목록
DEFAULT_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet (Google)
    "AMZN",  # Amazon
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "JNJ",   # Johnson & Johnson
]

# 기본 데이터 수집 기간
DEFAULT_PERIOD = "1y"  # 1년
DEFAULT_INTERVAL = "1d"  # 일봉 