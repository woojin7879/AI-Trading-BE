import pandas as pd
import numpy as np
import time
import schedule
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.config.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    TRADING_MODE, MAX_POSITION_SIZE, RISK_PER_TRADE, 
    STOP_LOSS_PERCENTAGE, setup_logging
)
from src.strategies.ml_strategy import MLTradingStrategy
from src.data.data_collector import YahooFinanceDataCollector

logger = setup_logging("trader")

class AlpacaTrader:
    """Alpaca API를 이용한 트레이딩 클래스"""
    
    def __init__(self, symbols=None, strategy_type="ml", paper=True):
        """
        초기화 함수
        
        Args:
            symbols (list): 거래할 주식 심볼 목록
            strategy_type (str): 전략 유형 ('ml' 또는 다른 전략)
            paper (bool): 페이퍼 트레이딩 여부
        """
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL"]
        self.strategy_type = strategy_type
        self.paper = paper
        
        # API 키 확인
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # Alpaca 클라이언트 초기화
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=paper)
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # 계좌 정보 가져오기
        self.account = self.trading_client.get_account()
        logger.info(f"계좌 정보: {self.account}")
        
        # 전략 초기화
        self.strategies = {}
        for symbol in self.symbols:
            if strategy_type == "ml":
                self.strategies[symbol] = MLTradingStrategy(
                    symbol=symbol,
                    risk_per_trade=RISK_PER_TRADE,
                    stop_loss_pct=STOP_LOSS_PERCENTAGE
                )
        
        # 데이터 수집기 초기화
        self.data_collector = YahooFinanceDataCollector(symbols=self.symbols)
        
        # 거래 기록 저장 디렉토리
        self.trades_dir = Path("data/trades")
        self.trades_dir.mkdir(exist_ok=True, parents=True)
    
    def get_account_info(self):
        """
        계좌 정보 가져오기
        
        Returns:
            dict: 계좌 정보
        """
        self.account = self.trading_client.get_account()
        
        account_info = {
            'cash': float(self.account.cash),
            'equity': float(self.account.equity),
            'buying_power': float(self.account.buying_power),
            'day_trade_count': self.account.daytrade_count,
            'status': self.account.status
        }
        
        return account_info
    
    def get_positions(self):
        """
        현재 포지션 가져오기
        
        Returns:
            dict: 심볼별 포지션 정보
        """
        positions = {}
        
        try:
            all_positions = self.trading_client.get_all_positions()
            
            for position in all_positions:
                symbol = position.symbol
                positions[symbol] = {
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price)
                }
            
            return positions
        except Exception as e:
            logger.error(f"포지션 정보 가져오기 실패: {e}")
            return {}
    
    def get_historical_data(self, symbol, timeframe=TimeFrame.Day, limit=100):
        """
        과거 데이터 가져오기
        
        Args:
            symbol (str): 주식 심볼
            timeframe (TimeFrame): 시간 프레임
            limit (int): 가져올 데이터 수
            
        Returns:
            pd.DataFrame: 과거 데이터
        """
        try:
            # 현재 시간
            end_time = datetime.now()
            
            # 시작 시간 (limit에 따라 계산)
            if timeframe == TimeFrame.Day:
                start_time = end_time - timedelta(days=limit)
            elif timeframe == TimeFrame.Hour:
                start_time = end_time - timedelta(hours=limit)
            else:
                start_time = end_time - timedelta(minutes=limit)
            
            # 데이터 요청
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            # 데이터프레임으로 변환
            if symbol in bars:
                df = pd.DataFrame(bars[symbol])
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'timestamp': 'Date'
                }, inplace=True)
                
                return df
            else:
                logger.warning(f"{symbol} 데이터를 찾을 수 없습니다.")
                return None
        except Exception as e:
            logger.error(f"{symbol} 과거 데이터 가져오기 실패: {e}")
            return None
    
    def place_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        """
        주문 실행
        
        Args:
            symbol (str): 주식 심볼
            qty (float): 수량
            side (str): 매수/매도 ('buy' 또는 'sell')
            order_type (str): 주문 유형 ('market', 'limit' 등)
            time_in_force (str): 주문 유효 기간 ('day', 'gtc' 등)
            
        Returns:
            dict: 주문 정보
        """
        try:
            # 주문 측 설정
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # 시간 유효성 설정
            if time_in_force.lower() == 'day':
                tif = TimeInForce.DAY
            else:
                tif = TimeInForce.GTC
            
            # 주문 요청 생성
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif
            )
            
            # 주문 실행
            order = self.trading_client.submit_order(order_request)
            
            # 주문 정보 반환
            order_info = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'type': order.type.value,
                'status': order.status.value,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
            }
            
            logger.info(f"주문 실행: {order_info}")
            
            # 주문 기록 저장
            self.save_trade(order_info)
            
            return order_info
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return None
    
    def cancel_all_orders(self):
        """
        모든 주문 취소
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.trading_client.cancel_orders()
            logger.info("모든 주문 취소 완료")
            return True
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return False
    
    def get_orders(self, status=None, limit=50):
        """
        주문 목록 가져오기
        
        Args:
            status (OrderStatus): 주문 상태
            limit (int): 가져올 주문 수
            
        Returns:
            list: 주문 목록
        """
        try:
            # 요청 파라미터 설정
            request_params = GetOrdersRequest(
                status=status,
                limit=limit
            )
            
            # 주문 가져오기
            orders = self.trading_client.get_orders(request_params)
            
            # 주문 정보 변환
            orders_info = []
            for order in orders:
                order_info = {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side.value,
                    'type': order.type.value,
                    'status': order.status.value,
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
                }
                orders_info.append(order_info)
            
            return orders_info
        except Exception as e:
            logger.error(f"주문 목록 가져오기 실패: {e}")
            return []
    
    def save_trade(self, trade_info):
        """
        거래 기록 저장
        
        Args:
            trade_info (dict): 거래 정보
        """
        try:
            # 현재 날짜로 파일명 생성
            date_str = datetime.now().strftime("%Y%m%d")
            file_path = self.trades_dir / f"trades_{date_str}.json"
            
            # 기존 파일이 있으면 로드
            trades = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    trades = json.load(f)
            
            # 거래 정보에 타임스탬프 추가
            trade_info['timestamp'] = datetime.now().isoformat()
            
            # 거래 정보 추가
            trades.append(trade_info)
            
            # 파일 저장
            with open(file_path, 'w') as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"거래 기록 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"거래 기록 저장 실패: {e}")
    
    def update_data(self):
        """
        데이터 업데이트
        """
        try:
            logger.info("데이터 업데이트 시작")
            self.data_collector.update_data()
            logger.info("데이터 업데이트 완료")
        except Exception as e:
            logger.error(f"데이터 업데이트 실패: {e}")
    
    def run_strategy(self, symbol):
        """
        단일 심볼에 대한 전략 실행
        
        Args:
            symbol (str): 주식 심볼
            
        Returns:
            dict: 매매 신호
        """
        try:
            # 전략 가져오기
            strategy = self.strategies.get(symbol)
            
            if strategy is None:
                logger.error(f"{symbol} 전략을 찾을 수 없습니다.")
                return None
            
            # 야후 파이낸스에서 최신 데이터 가져오기
            latest_data = self.data_collector.get_latest_data(symbol, days=100)
            
            if latest_data is None or latest_data.empty:
                logger.error(f"{symbol} 데이터를 가져올 수 없습니다.")
                return None
            
            # 계좌 정보 가져오기
            account_info = self.get_account_info()
            buying_power = float(account_info['buying_power'])
            
            # 매매 신호 생성
            signal = strategy.generate_signal(latest_data, buying_power)
            
            logger.info(f"{symbol} 매매 신호: {signal}")
            
            return signal
        except Exception as e:
            logger.error(f"{symbol} 전략 실행 실패: {e}")
            return None
    
    def execute_signal(self, symbol, signal):
        """
        매매 신호 실행
        
        Args:
            symbol (str): 주식 심볼
            signal (dict): 매매 신호
            
        Returns:
            dict: 주문 정보
        """
        if signal is None:
            return None
        
        action = signal.get('action')
        
        # 매수 신호
        if action == 'BUY':
            size = signal.get('size', 0)
            
            if size <= 0:
                logger.warning(f"{symbol} 매수 수량이 0 이하입니다.")
                return None
            
            # 주문 실행
            return self.place_order(symbol, size, 'buy')
        
        # 매도 신호
        elif action == 'SELL':
            # 현재 포지션 확인
            positions = self.get_positions()
            position = positions.get(symbol)
            
            if position is None:
                logger.warning(f"{symbol} 포지션이 없습니다.")
                return None
            
            size = position.get('qty', 0)
            
            if size <= 0:
                logger.warning(f"{symbol} 매도 수량이 0 이하입니다.")
                return None
            
            # 주문 실행
            return self.place_order(symbol, size, 'sell')
        
        # 홀드 신호
        else:
            logger.info(f"{symbol} 홀드 신호: {signal.get('reason')}")
            return None
    
    def run_trading_cycle(self):
        """
        전체 트레이딩 사이클 실행
        """
        try:
            logger.info("트레이딩 사이클 시작")
            
            # 데이터 업데이트
            self.update_data()
            
            # 계좌 정보 로깅
            account_info = self.get_account_info()
            logger.info(f"계좌 정보: 현금=${account_info['cash']}, 자산=${account_info['equity']}")
            
            # 현재 포지션 로깅
            positions = self.get_positions()
            logger.info(f"현재 포지션: {positions}")
            
            # 각 심볼에 대한 전략 실행
            for symbol in self.symbols:
                # 매매 신호 생성
                signal = self.run_strategy(symbol)
                
                if signal:
                    # 매매 신호 실행
                    order_info = self.execute_signal(symbol, signal)
                    
                    if order_info:
                        logger.info(f"{symbol} 주문 실행 완료: {order_info}")
                    else:
                        logger.info(f"{symbol} 주문 실행 없음")
            
            logger.info("트레이딩 사이클 완료")
        except Exception as e:
            logger.error(f"트레이딩 사이클 실행 실패: {e}")
    
    def schedule_trading(self, interval_minutes=60):
        """
        정기적인 트레이딩 스케줄 설정
        
        Args:
            interval_minutes (int): 트레이딩 간격 (분)
        """
        # 즉시 한 번 실행
        self.run_trading_cycle()
        
        # 정기적으로 실행
        schedule.every(interval_minutes).minutes.do(self.run_trading_cycle)
        
        logger.info(f"{interval_minutes}분 간격으로 트레이딩 스케줄 설정 완료")
        
        # 스케줄 실행
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("사용자에 의해 트레이딩 중단")
                break
            except Exception as e:
                logger.error(f"스케줄 실행 중 오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도

if __name__ == "__main__":
    # 트레이더 생성
    trader = AlpacaTrader(symbols=["AAPL", "MSFT", "GOOGL"], paper=True)
    
    # 트레이딩 스케줄 설정 (60분 간격)
    trader.schedule_trading(interval_minutes=60) 