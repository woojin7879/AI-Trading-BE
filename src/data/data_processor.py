import pandas as pd
import numpy as np
import ta
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from src.config.config import DATA_DIR, setup_logging

logger = setup_logging("data_processor")

class DataProcessor:
    """주식 데이터 전처리 클래스"""
    
    def __init__(self, input_dir=None, output_dir=None):
        """
        초기화 함수
        
        Args:
            input_dir (Path): 입력 데이터 디렉토리
            output_dir (Path): 출력 데이터 디렉토리
        """
        self.input_dir = input_dir or DATA_DIR / "raw"
        self.output_dir = output_dir or DATA_DIR / "processed"
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = None
        
    def load_data(self, symbol, interval="1d"):
        """
        데이터 로드
        
        Args:
            symbol (str): 주식 심볼
            interval (str): 데이터 간격
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        file_path = self.input_dir / f"{symbol}_{interval}.csv"
        
        if not file_path.exists():
            logger.error(f"파일을 찾을 수 없음: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """
        기술적 지표 추가
        
        Args:
            df (pd.DataFrame): 주가 데이터
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터
        """
        if df is None or df.empty:
            return df
        
        # 이동평균
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # 볼린저 밴드
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # 스토캐스틱 오실레이터
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # 가격 변화율
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_1D'] = df['Close'].pct_change(periods=1)
        df['Price_Change_5D'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20D'] = df['Close'].pct_change(periods=20)
        
        # 거래량 지표
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = ta.trend.sma_indicator(df['Volume'], window=5)
        df['Volume_MA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        # 추세 지표
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # NaN 값 처리
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def add_target_variable(self, df, days_ahead=1):
        """
        목표 변수 추가 (n일 후 가격 변화)
        
        Args:
            df (pd.DataFrame): 주가 데이터
            days_ahead (int): 예측할 미래 일수
            
        Returns:
            pd.DataFrame: 목표 변수가 추가된 데이터
        """
        if df is None or df.empty:
            return df
        
        # n일 후 종가
        df[f'Future_Close_{days_ahead}d'] = df['Close'].shift(-days_ahead)
        
        # n일 후 가격 변화율 (%)
        df[f'Target_{days_ahead}d'] = (df[f'Future_Close_{days_ahead}d'] - df['Close']) / df['Close'] * 100
        
        # 이진 목표 변수 (상승=1, 하락=0)
        df[f'Target_Binary_{days_ahead}d'] = (df[f'Target_{days_ahead}d'] > 0).astype(int)
        
        # NaN 값 제거 (마지막 n일은 목표 변수를 계산할 수 없음)
        df.dropna(subset=[f'Target_{days_ahead}d'], inplace=True)
        
        return df
    
    def normalize_features(self, df, feature_columns=None, scaler_type='minmax', fit=True, save_scaler=True, symbol=None):
        """
        특성 정규화
        
        Args:
            df (pd.DataFrame): 주가 데이터
            feature_columns (list): 정규화할 특성 컬럼 목록
            scaler_type (str): 스케일러 유형 ('minmax' 또는 'standard')
            fit (bool): 스케일러를 새로 학습할지 여부
            save_scaler (bool): 스케일러를 저장할지 여부
            symbol (str): 주식 심볼 (스케일러 저장 시 사용)
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        if df is None or df.empty:
            return df
        
        # 기본 특성 컬럼 (정규화할 컬럼)
        if feature_columns is None:
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                              'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200',
                              'BB_High', 'BB_Low', 'BB_Mid', 'BB_Width',
                              'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                              'Stoch_K', 'Stoch_D', 'ATR',
                              'Price_Change', 'Price_Change_1D', 'Price_Change_5D', 'Price_Change_20D',
                              'Volume_Change', 'Volume_MA_5', 'Volume_MA_20', 'ADX']
        
        # 존재하는 컬럼만 선택
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        if not feature_columns:
            logger.warning("정규화할 특성 컬럼이 없습니다.")
            return df
        
        # 스케일러 선택
        if fit or self.scaler is None:
            if scaler_type.lower() == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
        
        # 특성 정규화
        features = df[feature_columns].values
        
        if fit:
            normalized_features = self.scaler.fit_transform(features)
        else:
            normalized_features = self.scaler.transform(features)
        
        # 정규화된 특성을 데이터프레임에 추가
        for i, col in enumerate(feature_columns):
            df[f'{col}_Norm'] = normalized_features[:, i]
        
        # 스케일러 저장
        if save_scaler and symbol is not None:
            scaler_dir = self.output_dir / "scalers"
            scaler_dir.mkdir(exist_ok=True)
            scaler_path = scaler_dir / f"{symbol}_{scaler_type}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"스케일러 저장 완료: {scaler_path}")
        
        return df
    
    def load_scaler(self, symbol, scaler_type='minmax'):
        """
        저장된 스케일러 로드
        
        Args:
            symbol (str): 주식 심볼
            scaler_type (str): 스케일러 유형
            
        Returns:
            scaler: 로드된 스케일러
        """
        scaler_dir = self.output_dir / "scalers"
        scaler_path = scaler_dir / f"{symbol}_{scaler_type}_scaler.joblib"
        
        if not scaler_path.exists():
            logger.error(f"스케일러 파일을 찾을 수 없음: {scaler_path}")
            return None
        
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"스케일러 로드 완료: {scaler_path}")
            return self.scaler
        except Exception as e:
            logger.error(f"스케일러 로드 중 오류 발생: {e}")
            return None
    
    def prepare_data(self, symbol, interval="1d", days_ahead=1, train_size=0.8):
        """
        모델 학습을 위한 데이터 준비
        
        Args:
            symbol (str): 주식 심볼
            interval (str): 데이터 간격
            days_ahead (int): 예측할 미래 일수
            train_size (float): 학습 데이터 비율
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) 학습 및 테스트 데이터
        """
        # 데이터 로드
        df = self.load_data(symbol, interval)
        if df is None:
            return None, None, None, None
        
        # 기술적 지표 추가
        df = self.add_technical_indicators(df)
        
        # 목표 변수 추가
        df = self.add_target_variable(df, days_ahead)
        
        # 특성 정규화
        df = self.normalize_features(df, fit=True, save_scaler=True, symbol=symbol)
        
        # 학습에 사용할 특성 선택
        feature_columns = [col for col in df.columns if col.endswith('_Norm')]
        
        # 목표 변수 선택
        target_column = f'Target_Binary_{days_ahead}d'  # 이진 분류 목표
        
        # 학습 및 테스트 데이터 분할
        train_size = int(len(df) * train_size)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        X_train = train_data[feature_columns].values
        y_train = train_data[target_column].values
        
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values
        
        # 처리된 데이터 저장
        processed_dir = self.output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        df.to_csv(processed_dir / f"{symbol}_{interval}_processed.csv", index=False)
        
        return X_train, X_test, y_train, y_test
    
    def process_all_symbols(self, symbols, interval="1d", days_ahead=1):
        """
        여러 심볼에 대한 데이터 처리
        
        Args:
            symbols (list): 처리할 주식 심볼 목록
            interval (str): 데이터 간격
            days_ahead (int): 예측할 미래 일수
        """
        for symbol in symbols:
            logger.info(f"{symbol} 데이터 처리 중...")
            
            # 데이터 로드
            df = self.load_data(symbol, interval)
            if df is None:
                continue
            
            # 기술적 지표 추가
            df = self.add_technical_indicators(df)
            
            # 목표 변수 추가
            df = self.add_target_variable(df, days_ahead)
            
            # 특성 정규화
            df = self.normalize_features(df, fit=True, save_scaler=True, symbol=symbol)
            
            # 처리된 데이터 저장
            processed_dir = self.output_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            df.to_csv(processed_dir / f"{symbol}_{interval}_processed.csv", index=False)
            
            logger.info(f"{symbol} 데이터 처리 완료")

if __name__ == "__main__":
    from src.config.config import DEFAULT_SYMBOLS
    
    # 데이터 처리기 생성
    processor = DataProcessor()
    
    # 모든 심볼에 대한 데이터 처리
    processor.process_all_symbols(DEFAULT_SYMBOLS) 