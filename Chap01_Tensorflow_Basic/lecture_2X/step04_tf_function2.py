"""
- Tensorflow 2.0 특징
3. @tf.function 함수 장식자(데코레이터)
    - 여러 함수를 포함하는 main 함수
"""
import tensorflow as tf

# model 생성 함수
def linear_model(x):
    return x * 2 + 0.2  # 회귀 방정식

# model 오차 함수
def model_error(y, y_pred):
    return y - y_pred  # 오차

# model 평가 함수 : main
@tf.function
def model_evaluation(x, y):
    y_pred = linear_model(x)  # 함수 호출
    err = model_error(y, y_pred)  # 함수 호출
    return tf.reduce_mean(tf.square(err))

# x, y data 생성
X = tf.constant([1,2,3], dtype=tf.float32)
Y = tf.constant([2,4,6], dtype=tf.float32)
MSE = model_evaluation(X, Y)
print(f"MSE = {format(MSE,'.5f')}")
