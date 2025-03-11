import argparse
import logging
import os
from pathlib import Path
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.config import setup_logging, DEFAULT_SYMBOLS
from src.data.data_collector import YahooFinanceDataCollector
from src.data.data_processor import DataProcessor
from src.models.model_builder import ModelBuilder
from src.strategies.ml_strategy import MLTradingStrategy
from src.trading.trader import AlpacaTrader

# 로거 설정
logger = setup_logging("main")

def collect_data(symbols=None, update=False):
    """
    데이터 수집
    
    Args:
        symbols (list): 수집할 주식 심볼 목록
        update (bool): 업데이트 여부
    """
    symbols = symbols or DEFAULT_SYMBOLS
    collector = YahooFinanceDataCollector(symbols=symbols)
    
    if update:
        logger.info("데이터 업데이트 시작")
        collector.update_data()
    else:
        logger.info("데이터 다운로드 시작")
        collector.download_all()
    
    logger.info("데이터 수집 완료")

def process_data(symbols=None, days_ahead=1):
    """
    데이터 처리
    
    Args:
        symbols (list): 처리할 주식 심볼 목록
        days_ahead (int): 예측할 미래 일수
    """
    symbols = symbols or DEFAULT_SYMBOLS
    processor = DataProcessor()
    
    logger.info("데이터 처리 시작")
    processor.process_all_symbols(symbols, days_ahead=days_ahead)
    logger.info("데이터 처리 완료")

def train_models(symbols=None, model_type="dense", epochs=100, batch_size=32, days_ahead=1):
    """
    모델 학습
    
    Args:
        symbols (list): 학습할 주식 심볼 목록
        model_type (str): 모델 유형
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        days_ahead (int): 예측할 미래 일수
    """
    symbols = symbols or DEFAULT_SYMBOLS
    
    for symbol in symbols:
        logger.info(f"{symbol} 모델 학습 시작")
        
        strategy = MLTradingStrategy(
            symbol=symbol,
            model_type=model_type
        )
        
        metrics = strategy.train_model(
            days_ahead=days_ahead,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if metrics:
            logger.info(f"{symbol} 모델 학습 완료: 정확도={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        else:
            logger.error(f"{symbol} 모델 학습 실패")

def backtest(symbols=None, start_date=None, end_date=None, initial_balance=10000):
    """
    백테스트 수행
    
    Args:
        symbols (list): 백테스트할 주식 심볼 목록
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        initial_balance (float): 초기 잔고
    """
    symbols = symbols or DEFAULT_SYMBOLS
    
    for symbol in symbols:
        logger.info(f"{symbol} 백테스트 시작")
        
        strategy = MLTradingStrategy(symbol=symbol)
        
        results = strategy.backtest(
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance
        )
        
        if results is not None and not results.empty:
            # 결과 저장
            output_dir = Path("data/backtest")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            output_file = output_dir / f"{symbol}_backtest_results.csv"
            results.to_csv(output_file, index=False)
            
            logger.info(f"{symbol} 백테스트 결과 저장 완료: {output_file}")
        else:
            logger.warning(f"{symbol} 백테스트 결과 없음")

def start_trading(symbols=None, paper=True, interval_minutes=60):
    """
    트레이딩 시작
    
    Args:
        symbols (list): 거래할 주식 심볼 목록
        paper (bool): 페이퍼 트레이딩 여부
        interval_minutes (int): 트레이딩 간격 (분)
    """
    symbols = symbols or DEFAULT_SYMBOLS
    
    logger.info(f"트레이딩 시작: 심볼={symbols}, 페이퍼={paper}, 간격={interval_minutes}분")
    
    trader = AlpacaTrader(
        symbols=symbols,
        paper=paper
    )
    
    trader.schedule_trading(interval_minutes=interval_minutes)

def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description="야후 파이낸스 기반 AI 주식 자동 매매 시스템")
    
    # 서브 커맨드 설정
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령")
    
    # 데이터 수집 커맨드
    collect_parser = subparsers.add_parser("collect", help="데이터 수집")
    collect_parser.add_argument("--symbols", nargs="+", help="수집할 주식 심볼 목록")
    collect_parser.add_argument("--update", action="store_true", help="기존 데이터 업데이트")
    
    # 데이터 처리 커맨드
    process_parser = subparsers.add_parser("process", help="데이터 처리")
    process_parser.add_argument("--symbols", nargs="+", help="처리할 주식 심볼 목록")
    process_parser.add_argument("--days-ahead", type=int, default=1, help="예측할 미래 일수")
    
    # 모델 학습 커맨드
    train_parser = subparsers.add_parser("train", help="모델 학습")
    train_parser.add_argument("--symbols", nargs="+", help="학습할 주식 심볼 목록")
    train_parser.add_argument("--model-type", choices=["dense", "lstm", "ensemble"], default="dense", help="모델 유형")
    train_parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")
    train_parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    train_parser.add_argument("--days-ahead", type=int, default=1, help="예측할 미래 일수")
    
    # 백테스트 커맨드
    backtest_parser = subparsers.add_parser("backtest", help="백테스트 수행")
    backtest_parser.add_argument("--symbols", nargs="+", help="백테스트할 주식 심볼 목록")
    backtest_parser.add_argument("--start-date", help="시작 날짜 (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", help="종료 날짜 (YYYY-MM-DD)")
    backtest_parser.add_argument("--initial-balance", type=float, default=10000, help="초기 잔고")
    
    # 트레이딩 커맨드
    trade_parser = subparsers.add_parser("trade", help="트레이딩 시작")
    trade_parser.add_argument("--symbols", nargs="+", help="거래할 주식 심볼 목록")
    trade_parser.add_argument("--live", action="store_true", help="실제 거래 (기본값: 페이퍼 트레이딩)")
    trade_parser.add_argument("--interval", type=int, default=60, help="트레이딩 간격 (분)")
    
    # 전체 파이프라인 커맨드
    pipeline_parser = subparsers.add_parser("pipeline", help="전체 파이프라인 실행 (수집 -> 처리 -> 학습 -> 백테스트)")
    pipeline_parser.add_argument("--symbols", nargs="+", help="처리할 주식 심볼 목록")
    pipeline_parser.add_argument("--model-type", choices=["dense", "lstm", "ensemble"], default="dense", help="모델 유형")
    pipeline_parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if hasattr(args, "symbols") and args.symbols else DEFAULT_SYMBOLS
    
    # 명령 실행
    if args.command == "collect":
        collect_data(symbols=symbols, update=args.update)
    
    elif args.command == "process":
        process_data(symbols=symbols, days_ahead=args.days_ahead)
    
    elif args.command == "train":
        train_models(
            symbols=symbols,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            days_ahead=args.days_ahead
        )
    
    elif args.command == "backtest":
        backtest(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.initial_balance
        )
    
    elif args.command == "trade":
        start_trading(
            symbols=symbols,
            paper=not args.live,
            interval_minutes=args.interval
        )
    
    elif args.command == "pipeline":
        # 전체 파이프라인 실행
        collect_data(symbols=symbols)
        process_data(symbols=symbols)
        train_models(
            symbols=symbols,
            model_type=args.model_type,
            epochs=args.epochs
        )
        backtest(symbols=symbols)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 