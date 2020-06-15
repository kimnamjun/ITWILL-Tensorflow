"""
- X(1) -> Y
- 손실함수(loss function) : 오차 반환함수
- 모델 최적화 알고리즘 : 경사하강법 알고리즘(GD, Adam) 적용
    -> 모델 학습 : 최적의 기울기, 절편 -> loss 값이 0에 수렴
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# X, Y data 정의
x_data = np.array([1,2,3])  # 입력 data
y_data = np.array([2,4,6])  # 출력 data

# X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None])  # x_data 공급
Y = tf.placeholder(dtype=tf.float32, shape=[None])  # y_data 공급

# a, b 변수 선언
a = tf.Variable(tf.random_normal([1]))  # 기울기
b = tf.Variable(tf.random_normal([1]))  # 절편

# 식 정의
model = tf.multiply(X, a) + b  # 예측치 회귀방정식
err = Y - model  # 오차
loss = tf.reduce_mean(tf.square(err))  # 손실 함수

# 최적화 객체
optimizer = tf.train.GradientDescentOptimizer(0.1)  # 학습률=0.1
train = optimizer.minimize(loss)  # 손실 최소화 : 최적의 기울기, 절편 수정

# 반복학습
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 변수 초기화 : a, b
    a_val, b_val = sess.run([a, b])
    print('최초 기울기와 절편')
    print('a = {}, b = {}'.format(a_val, b_val))

    feed_data = {X: x_data, Y:y_data}

    # 반복 학습 : 50회
    for step in range(50):
        _, loss_val = sess.run([train, loss], feed_dict=feed_data)
        a_val, b_val = sess.run([a, b])
        print(f'step = {step+1}, loss = {loss_val}, a = {a_val}, b = {b_val}')
