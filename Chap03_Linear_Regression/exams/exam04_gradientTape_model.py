# -*- coding: utf-8 -*-

'''
문) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성하기 
  조건1> train/test - 7:3비율 
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> learning_rate=0.005
  조건5> optimizer = tf.keras.optimizers.Adam
  조건6> epoch(step) = 5000회
  조건7> 모델 평가 : MSE
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale # 정규화
from sklearn.metrics import mean_squared_error
import tensorflow as tf 
import numpy as np

# 1. data load
boston = load_boston()
print(boston) # "data", "target"

# 변수 선택  
X = boston.data  
y = boston.target
X.shape # (506, 13)

y_nor = minmax_scale(y) # 정규화 

# train/test split(90 vs 10)
x_train, x_test, y_train, y_test = train_test_split(
        X, y_nor, test_size=0.1, random_state=123)

tf.random.set_seed(123)

# 2. Model 클래스 : model = input * w + b
   

# 3. 손실함수 : (예측치, 정답) -> 오차 


# 4. 기울기 계산 함수 : 오차값 -> 기울기 반환 


# 5. 모델 및 최적화 객체   


# 6. 반복 학습 : Model 객체와 손실함수 이용

    
# 7. 최적화된 model 
