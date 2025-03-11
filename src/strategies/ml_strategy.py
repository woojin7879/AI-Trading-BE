import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

from src.config.config import setup_logging
from src.models.model_builder import ModelBuilder
from src.data.data_processor import DataProcessor

logger = setup_logging("ml_strategy")

class MLTradingStrategy:
    """머신러닝 기반 트레이딩 전략 클래스"""
    
    def __init__(self, symbol, model_name=None, model_type="dense", risk_per_trade=0.02, stop_loss_pct=0.02):
        """
        초기화 함수
        
        Args:
            symbol (str): 주식 심볼
            model_name (str): 모델 이름
            model_type (str): 모델 유형
            risk_per_trade (float): 거래당 리스크 (계좌의 %)
            stop_loss_pct (float): 손절매 비율
        """
        self.symbol = symbol
        self.model_name = model_name or f"{symbol}_predictor"
        self.model_type = model_type
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        
        # 데이터 처리기 및 모델 빌더 초기화
        self.data_processor = DataProcessor()
        self.model_builder = ModelBuilder(model_name=self.model_name, model_type=self.model_type)
        
        # 모델 로드 시도
        self.model = self.model_builder.load_model()
        
        # 현재 포지션 상태
        self.position = None
        self.entry_price = None
        self.stop_loss_price = None
        self.position_size = 0
    
    def train_model(self, days_ahead=1, epochs=100, batch_size=32):
        """
        모델 학습
        
        Args:
            days_ahead (int): 예측할 미래 일수
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            
        Returns:
            dict: 평가 지표
        """
        logger.info(f"{self.symbol} 모델 학습 시작")
        
        # 데이터 준비
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(
            self.symbol, days_ahead=days_ahead
        )
        
        if X_train is None or y_train is None:
            logger.error(f"{self.symbol} 학습 데이터 준비 실패")
            return None
        
        # 모델 구축
        input_shape = X_train.shape[1:]
        self.model_builder.build_model(input_shape)
        
        # 모델 학습
        history = self.model_builder.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # 학습 히스토리 시각화
        self.model_builder.plot_training_history(history)
        
        # 모델 평가
        metrics = self.model_builder.evaluate(X_test, y_test)
        
        # 모델 업데이트
        self.model = self.model_builder.model
        
        logger.info(f"{self.symbol} 모델 학습 완료")
        
        return metrics
    
    def prepare_features(self, data):
        """
        예측을 위한 특성 준비
        
        Args:
            data (pd.DataFrame): 주가 데이터
            
        Returns:
            np.array: 예측을 위한 특성
        """
        # 기술적 지표 추가
        df = self.data_processor.add_technical_indicators(data.copy())
        
        # 스케일러 로드
        scaler = self.data_processor.load_scaler(self.symbol)
        
        if scaler is None:
            logger.warning(f"{self.symbol} 스케일러를 찾을 수 없어 새로 생성합니다.")
            # 특성 정규화 (새 스케일러 학습)
            df = self.data_processor.normalize_features(df, fit=True, save_scaler=True, symbol=self.symbol)
        else:
            # 특성 정규화 (기존 스케일러 사용)
            df = self.data_processor.normalize_features(df, fit=False, save_scaler=False)
        
        # 예측에 사용할 특성 선택
        feature_columns = [col for col in df.columns if col.endswith('_Norm')]
        
        if not feature_columns:
            logger.error("예측에 사용할 특성이 없습니다.")
            return None
        
        # 특성 배열 반환
        return df[feature_columns].values
    
    def predict(self, data):
        """
        주가 방향 예측
        
        Args:
            data (pd.DataFrame): 주가 데이터
            
        Returns:
            tuple: (예측 결과, 예측 확률)
        """
        if self.model is None:
            logger.error("예측할 모델이 없습니다.")
            return None, None
        
        # 특성 준비
        features = self.prepare_features(data)
        
        if features is None:
            return None, None
        
        # 예측 수행
        y_pred, y_pred_prob = self.model_builder.predict(features)
        
        return y_pred, y_pred_prob
    
    def calculate_position_size(self, account_balance, current_price):
        """
        포지션 크기 계산
        
        Args:
            account_balance (float): 계좌 잔고
            current_price (float): 현재 가격
            
        Returns:
            int: 매수할 주식 수량
        """
        # 거래당 리스크 금액
        risk_amount = account_balance * self.risk_per_trade
        
        # 주당 리스크 (손절매 기준)
        per_share_risk = current_price * self.stop_loss_pct
        
        # 매수 가능 수량
        shares = int(risk_amount / per_share_risk)
        
        # 총 매수 금액이 계좌 잔고의 절반을 넘지 않도록 제한
        max_shares = int((account_balance * 0.5) / current_price)
        shares = min(shares, max_shares)
        
        return shares
    
    def generate_signal(self, data, account_balance):
        """
        매매 신호 생성
        
        Args:
            data (pd.DataFrame): 주가 데이터
            account_balance (float): 계좌 잔고
            
        Returns:
            dict: 매매 신호
        """
        # 현재 가격
        current_price = data['Close'].iloc[-1]
        
        # 예측 수행
        y_pred, y_pred_prob = self.predict(data)
        
        if y_pred is None:
            return {
                'action': 'HOLD',
                'reason': '예측 실패',
                'confidence': 0.0,
                'price': current_price
            }
        
        # 마지막 행의 예측 결과
        prediction = y_pred[-1][0]
        confidence = y_pred_prob[-1][0]
        
        # 현재 포지션이 없는 경우
        if self.position is None:
            # 상승 예측 (매수 신호)
            if prediction == 1 and confidence >= 0.6:
                # 포지션 크기 계산
                self.position_size = self.calculate_position_size(account_balance, current_price)
                
                # 매수 가격 및 손절매 가격 설정
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                
                # 포지션 상태 업데이트
                self.position = 'LONG'
                
                return {
                    'action': 'BUY',
                    'reason': f'상승 예측 (확률: {confidence:.2f})',
                    'confidence': confidence,
                    'price': current_price,
                    'size': self.position_size,
                    'stop_loss': self.stop_loss_price
                }
            else:
                return {
                    'action': 'HOLD',
                    'reason': f'매수 조건 불충족 (예측: {"상승" if prediction == 1 else "하락"}, 확률: {confidence:.2f})',
                    'confidence': confidence,
                    'price': current_price
                }
        
        # 롱 포지션이 있는 경우
        elif self.position == 'LONG':
            # 손절매 조건 확인
            if current_price <= self.stop_loss_price:
                # 포지션 정리
                self.position = None
                self.entry_price = None
                self.stop_loss_price = None
                
                return {
                    'action': 'SELL',
                    'reason': '손절매',
                    'confidence': confidence,
                    'price': current_price,
                    'size': self.position_size,
                    'profit_pct': (current_price - self.entry_price) / self.entry_price * 100
                }
            
            # 하락 예측 (매도 신호)
            if prediction == 0 and confidence >= 0.6:
                # 수익률 계산
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                
                # 포지션 정리
                position_size = self.position_size
                self.position = None
                self.entry_price = None
                self.stop_loss_price = None
                self.position_size = 0
                
                return {
                    'action': 'SELL',
                    'reason': f'하락 예측 (확률: {confidence:.2f})',
                    'confidence': confidence,
                    'price': current_price,
                    'size': position_size,
                    'profit_pct': profit_pct
                }
            else:
                return {
                    'action': 'HOLD',
                    'reason': f'매도 조건 불충족 (예측: {"상승" if prediction == 1 else "하락"}, 확률: {confidence:.2f})',
                    'confidence': confidence,
                    'price': current_price,
                    'unrealized_profit_pct': (current_price - self.entry_price) / self.entry_price * 100
                }
    
    def backtest(self, start_date=None, end_date=None, initial_balance=10000):
        """
        백테스트 수행
        
        Args:
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            initial_balance (float): 초기 잔고
            
        Returns:
            pd.DataFrame: 백테스트 결과
        """
        logger.info(f"{self.symbol} 백테스트 시작")
        
        # 데이터 로드
        df = self.data_processor.load_data(self.symbol)
        
        if df is None:
            logger.error(f"{self.symbol} 데이터 로드 실패")
            return None
        
        # 날짜 필터링
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # 기술적 지표 추가
        df = self.data_processor.add_technical_indicators(df)
        
        # 백테스트 결과 저장용 데이터프레임
        results = []
        
        # 초기 설정
        balance = initial_balance
        position = None
        entry_price = None
        position_size = 0
        
        # 백테스트 수행
        for i in range(50, len(df)):  # 충분한 데이터가 있는 시점부터 시작
            # 현재 날짜까지의 데이터
            current_data = df.iloc[:i+1].copy()
            current_date = current_data['Date'].iloc[-1]
            current_price = current_data['Close'].iloc[-1]
            
            # 예측 수행
            features = self.prepare_features(current_data)
            
            if features is None or self.model is None:
                continue
            
            y_pred, y_pred_prob = self.model_builder.predict(features)
            
            if y_pred is None:
                continue
            
            # 마지막 행의 예측 결과
            prediction = y_pred[-1][0]
            confidence = y_pred_prob[-1][0]
            
            # 매매 신호 생성
            signal = None
            
            # 포지션이 없는 경우
            if position is None:
                # 상승 예측 (매수 신호)
                if prediction == 1 and confidence >= 0.6:
                    # 매수 가능 수량 계산
                    max_shares = int(balance / current_price)
                    position_size = min(max_shares, int(balance * 0.5 / current_price))
                    
                    if position_size > 0:
                        # 매수 실행
                        cost = position_size * current_price
                        balance -= cost
                        position = 'LONG'
                        entry_price = current_price
                        
                        signal = {
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'size': position_size,
                            'cost': cost,
                            'balance': balance,
                            'prediction': prediction,
                            'confidence': confidence
                        }
            
            # 롱 포지션이 있는 경우
            elif position == 'LONG':
                # 하락 예측 (매도 신호) 또는 손절매
                if prediction == 0 and confidence >= 0.6:
                    # 매도 실행
                    revenue = position_size * current_price
                    profit = revenue - (position_size * entry_price)
                    balance += revenue
                    
                    signal = {
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'size': position_size,
                        'revenue': revenue,
                        'profit': profit,
                        'profit_pct': (current_price - entry_price) / entry_price * 100,
                        'balance': balance,
                        'prediction': prediction,
                        'confidence': confidence
                    }
                    
                    # 포지션 정리
                    position = None
                    entry_price = None
                    position_size = 0
            
            # 신호가 있으면 결과에 추가
            if signal:
                results.append(signal)
        
        # 마지막 포지션 정리
        if position == 'LONG':
            current_price = df['Close'].iloc[-1]
            current_date = df['Date'].iloc[-1]
            
            # 매도 실행
            revenue = position_size * current_price
            profit = revenue - (position_size * entry_price)
            balance += revenue
            
            results.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'size': position_size,
                'revenue': revenue,
                'profit': profit,
                'profit_pct': (current_price - entry_price) / entry_price * 100,
                'balance': balance,
                'prediction': None,
                'confidence': None,
                'note': '백테스트 종료 시 포지션 정리'
            })
        
        # 결과 데이터프레임 생성
        if results:
            results_df = pd.DataFrame(results)
            
            # 성과 지표 계산
            total_trades = len(results_df)
            winning_trades = len(results_df[results_df['action'] == 'SELL'][results_df.get('profit', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            if total_trades > 0:
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0
            
            final_balance = balance
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            logger.info(f"{self.symbol} 백테스트 결과:")
            logger.info(f"총 거래 수: {total_trades}")
            logger.info(f"승률: {win_rate:.2%}")
            logger.info(f"최종 잔고: ${final_balance:.2f}")
            logger.info(f"총 수익률: {total_return:.2f}%")
            
            return results_df
        else:
            logger.warning(f"{self.symbol} 백테스트 결과 없음")
            return pd.DataFrame()

if __name__ == "__main__":
    # 전략 생성
    strategy = MLTradingStrategy(symbol="AAPL", model_type="dense")
    
    # 모델 학습
    strategy.train_model(epochs=50)
    
    # 백테스트
    results = strategy.backtest(initial_balance=10000)
    
    if results is not None and not results.empty:
        print(results.head()) 