import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
import os

from src.config.config import MODEL_DIR, setup_logging

logger = setup_logging("model_builder")

class ModelBuilder:
    """주식 예측 모델 구축 클래스"""
    
    def __init__(self, model_name="stock_predictor", model_type="lstm"):
        """
        초기화 함수
        
        Args:
            model_name (str): 모델 이름
            model_type (str): 모델 유형 ('lstm', 'dense', 'ensemble')
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.model_dir = MODEL_DIR
        self.model_dir.mkdir(exist_ok=True)
        
    def build_lstm_model(self, input_shape, output_shape=1):
        """
        LSTM 모델 구축
        
        Args:
            input_shape (tuple): 입력 데이터 형태
            output_shape (int): 출력 데이터 형태
            
        Returns:
            Model: 구축된 LSTM 모델
        """
        model = Sequential([
            LSTM(units=64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"LSTM 모델 구축 완료: {model.summary()}")
        self.model = model
        return model
    
    def build_dense_model(self, input_shape, output_shape=1):
        """
        Dense 모델 구축
        
        Args:
            input_shape (tuple): 입력 데이터 형태
            output_shape (int): 출력 데이터 형태
            
        Returns:
            Model: 구축된 Dense 모델
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Dense 모델 구축 완료: {model.summary()}")
        self.model = model
        return model
    
    def build_ensemble_model(self, input_shape, output_shape=1):
        """
        앙상블 모델 구축
        
        Args:
            input_shape (tuple): 입력 데이터 형태
            output_shape (int): 출력 데이터 형태
            
        Returns:
            Model: 구축된 앙상블 모델
        """
        # 입력 레이어
        input_layer = Input(shape=input_shape)
        
        # LSTM 브랜치
        lstm_branch = LSTM(units=64, return_sequences=True)(input_layer)
        lstm_branch = Dropout(0.2)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = LSTM(units=32, return_sequences=False)(lstm_branch)
        lstm_branch = Dropout(0.2)(lstm_branch)
        
        # Dense 브랜치
        dense_branch = Dense(128, activation='relu')(input_layer)
        dense_branch = Dropout(0.3)(dense_branch)
        dense_branch = BatchNormalization()(dense_branch)
        dense_branch = Dense(64, activation='relu')(dense_branch)
        dense_branch = Dropout(0.3)(dense_branch)
        
        # 브랜치 결합
        merged = tf.keras.layers.concatenate([lstm_branch, dense_branch])
        
        # 출력 레이어
        output = Dense(32, activation='relu')(merged)
        output = Dropout(0.2)(output)
        output = Dense(16, activation='relu')(output)
        output = Dense(output_shape, activation='sigmoid')(output)
        
        # 모델 생성
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"앙상블 모델 구축 완료: {model.summary()}")
        self.model = model
        return model
    
    def build_model(self, input_shape, output_shape=1):
        """
        모델 유형에 따른 모델 구축
        
        Args:
            input_shape (tuple): 입력 데이터 형태
            output_shape (int): 출력 데이터 형태
            
        Returns:
            Model: 구축된 모델
        """
        if self.model_type == 'lstm':
            return self.build_lstm_model(input_shape, output_shape)
        elif self.model_type == 'dense':
            return self.build_dense_model(input_shape, output_shape)
        elif self.model_type == 'ensemble':
            return self.build_ensemble_model(input_shape, output_shape)
        else:
            logger.error(f"지원되지 않는 모델 유형: {self.model_type}")
            raise ValueError(f"지원되지 않는 모델 유형: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, validation_split=0.2):
        """
        모델 학습
        
        Args:
            X_train (np.array): 학습 데이터
            y_train (np.array): 학습 레이블
            X_val (np.array): 검증 데이터
            y_val (np.array): 검증 레이블
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            validation_split (float): 검증 데이터 비율
            
        Returns:
            History: 학습 히스토리
        """
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.build_model(input_shape)
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=self.model_dir / f"{self.model_name}_best.h5",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # 검증 데이터가 제공되지 않은 경우 학습 데이터에서 분할
        if X_val is None or y_val is None:
            validation_data = None
            val_split = validation_split
        else:
            validation_data = (X_val, y_val)
            val_split = 0.0
        
        # 모델 학습
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        self.save_model()
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test (np.array): 테스트 데이터
            y_test (np.array): 테스트 레이블
            
        Returns:
            dict: 평가 지표
        """
        if self.model is None:
            logger.error("모델이 구축되지 않았습니다.")
            return None
        
        # 모델 예측
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        # 결과 출력
        logger.info(f"모델 평가 결과:")
        logger.info(f"정확도: {accuracy:.4f}")
        logger.info(f"정밀도: {precision:.4f}")
        logger.info(f"재현율: {recall:.4f}")
        logger.info(f"F1 점수: {f1:.4f}")
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.model_dir / f"{self.model_name}_confusion_matrix.png")
        
        # 평가 지표 반환
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def save_model(self):
        """
        모델 저장
        """
        if self.model is None:
            logger.error("저장할 모델이 없습니다.")
            return
        
        # 모델 저장 경로
        model_path = self.model_dir / f"{self.model_name}.h5"
        
        # 모델 저장
        self.model.save(model_path)
        logger.info(f"모델 저장 완료: {model_path}")
    
    def load_model(self, model_path=None):
        """
        모델 로드
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            Model: 로드된 모델
        """
        if model_path is None:
            model_path = self.model_dir / f"{self.model_name}.h5"
        
        if not os.path.exists(model_path):
            logger.error(f"모델 파일을 찾을 수 없음: {model_path}")
            return None
        
        try:
            self.model = load_model(model_path)
            logger.info(f"모델 로드 완료: {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            return None
    
    def predict(self, X):
        """
        예측 수행
        
        Args:
            X (np.array): 예측할 데이터
            
        Returns:
            np.array: 예측 결과
        """
        if self.model is None:
            logger.error("예측할 모델이 없습니다.")
            return None
        
        # 예측 수행
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        return y_pred, y_pred_prob
    
    def plot_training_history(self, history):
        """
        학습 히스토리 시각화
        
        Args:
            history (History): 학습 히스토리
        """
        # 손실 그래프
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 정확도 그래프
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / f"{self.model_name}_training_history.png")
        plt.close()

if __name__ == "__main__":
    # 예시 데이터 생성
    X = np.random.random((100, 30))
    y = np.random.randint(0, 2, 100)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 빌더 생성
    model_builder = ModelBuilder(model_name="test_model", model_type="dense")
    
    # 모델 구축
    model = model_builder.build_model(input_shape=(30,))
    
    # 모델 학습
    history = model_builder.train(X_train, y_train, epochs=50, batch_size=16)
    
    # 학습 히스토리 시각화
    model_builder.plot_training_history(history)
    
    # 모델 평가
    metrics = model_builder.evaluate(X_test, y_test)
    
    # 예측
    y_pred, y_pred_prob = model_builder.predict(X_test) 