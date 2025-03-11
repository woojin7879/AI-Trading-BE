import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm

from src.config.config import (
    DATA_DIR, DEFAULT_SYMBOLS, DEFAULT_PERIOD, 
    DEFAULT_INTERVAL, setup_logging
)

logger = setup_logging("data_collector")

class YahooFinanceDataCollector:
    """야후 파이낸스에서 주식 데이터를 수집하는 클래스"""
    
    def __init__(self, symbols=None, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
        """
        초기화 함수
        
        Args:
            symbols (list): 수집할 주식 심볼 목록
            period (str): 데이터 수집 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): 데이터 간격 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.period = period
        self.interval = interval
        self.data_dir = DATA_DIR / "raw"
        self.data_dir.mkdir(exist_ok=True)
        
    def download_data(self, symbol):
        """
        단일 심볼에 대한 데이터 다운로드
        
        Args:
            symbol (str): 주식 심볼
            
        Returns:
            pd.DataFrame: 다운로드된 데이터
        """
        try:
            logger.info(f"다운로드 중: {symbol}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            # 인덱스 리셋 및 날짜 컬럼 추가
            df.reset_index(inplace=True)
            
            # 심볼 컬럼 추가
            df['Symbol'] = symbol
            
            # NaN 값 처리
            df.fillna(method='ffill', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"{symbol} 다운로드 중 오류 발생: {e}")
            return None
    
    def download_all(self):
        """
        모든 심볼에 대한 데이터 다운로드 및 저장
        
        Returns:
            pd.DataFrame: 모든 심볼의 데이터를 포함하는 데이터프레임
        """
        all_data = []
        
        for symbol in tqdm(self.symbols, desc="데이터 다운로드 중"):
            df = self.download_data(symbol)
            if df is not None and not df.empty:
                all_data.append(df)
                
                # 개별 심볼 데이터 저장
                file_path = self.data_dir / f"{symbol}_{self.interval}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"{symbol} 데이터 저장 완료: {file_path}")
                
                # API 제한을 피하기 위한 지연
                time.sleep(1)
        
        if all_data:
            # 모든 데이터 결합
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 결합된 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d")
            file_path = self.data_dir / f"combined_{self.interval}_{timestamp}.csv"
            combined_data.to_csv(file_path, index=False)
            logger.info(f"결합된 데이터 저장 완료: {file_path}")
            
            return combined_data
        else:
            logger.warning("다운로드된 데이터가 없습니다.")
            return pd.DataFrame()
    
    def get_latest_data(self, symbol, days=30):
        """
        최근 n일 데이터 가져오기
        
        Args:
            symbol (str): 주식 심볼
            days (int): 가져올 일수
            
        Returns:
            pd.DataFrame: 최근 데이터
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=self.interval)
            
            # 인덱스 리셋 및 날짜 컬럼 추가
            df.reset_index(inplace=True)
            
            # 심볼 컬럼 추가
            df['Symbol'] = symbol
            
            return df
        except Exception as e:
            logger.error(f"{symbol} 최근 데이터 가져오기 중 오류 발생: {e}")
            return None
    
    def update_data(self):
        """
        기존 데이터 업데이트
        """
        for symbol in tqdm(self.symbols, desc="데이터 업데이트 중"):
            file_path = self.data_dir / f"{symbol}_{self.interval}.csv"
            
            if file_path.exists():
                # 기존 데이터 로드
                existing_data = pd.read_csv(file_path)
                
                # 최근 날짜 확인
                if 'Date' in existing_data.columns:
                    latest_date = pd.to_datetime(existing_data['Date']).max()
                    days_since = (datetime.now() - latest_date).days
                    
                    # 최근 데이터 가져오기
                    if days_since > 0:
                        new_data = self.get_latest_data(symbol, days=days_since+5)  # 겹치는 부분을 위해 여유 추가
                        
                        if new_data is not None and not new_data.empty:
                            # 중복 제거를 위해 기존 데이터의 최근 날짜 이후 데이터만 선택
                            new_data['Date'] = pd.to_datetime(new_data['Date'])
                            new_data = new_data[new_data['Date'] > latest_date]
                            
                            if not new_data.empty:
                                # 새 데이터 추가
                                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                                updated_data.to_csv(file_path, index=False)
                                logger.info(f"{symbol} 데이터 업데이트 완료: {len(new_data)}행 추가됨")
                            else:
                                logger.info(f"{symbol} 새로운 데이터 없음")
                    else:
                        logger.info(f"{symbol} 이미 최신 데이터")
            else:
                # 파일이 없으면 새로 다운로드
                df = self.download_data(symbol)
                if df is not None and not df.empty:
                    df.to_csv(file_path, index=False)
                    logger.info(f"{symbol} 데이터 새로 다운로드 완료")
                
                # API 제한을 피하기 위한 지연
                time.sleep(1)

if __name__ == "__main__":
    # 데이터 수집기 생성
    collector = YahooFinanceDataCollector()
    
    # 모든 데이터 다운로드
    collector.download_all()
    
    # 또는 데이터 업데이트
    # collector.update_data() 