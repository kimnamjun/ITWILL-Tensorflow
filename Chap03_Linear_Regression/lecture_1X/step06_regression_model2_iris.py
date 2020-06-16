"""
y변수 : 1 컬럼
x변수 : 2~4 컬럼

model 최적화 알고리즘 : GD -> Adam
model 평가 : MSE
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. 공급 데이터
iris = pd.read_csv('C:/ITWILL/6_Tensorflow/data/iris.csv')
cols = list(iris.columns)
x_data = iris[cols[1:4]]
y_data = iris[cols[0]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# 3. a(w), b 변수 정의 : 난수 초기값
a = tf.Variable(tf.random_normal(shape=[3, 1]))  # [입력수, 출력수]
b = tf.Variable(tf.random_normal(shape=[1]))

# 4. model 생성
model = tf.matmul(X, a) + b  # 예측치
loss = tf.reduce_mean(tf.square(Y - model))
opt = tf.train.AdamOptimizer(0.5)  # 학습률 = 0.5
train = opt.minimize(loss)  # 소실 최소화 식

# 5. model 학습 -> model 최적화(최적의 a, b 업데이트)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # a, b 초기화
    a_val, b_val = sess.run([a, b])
    print(f'최초 기울기 : {a_val}, 절편 : {b_val}')

    feed_data = {X: x_train, Y: y_train}  # 훈련용 공급 데이터

    for step in range(100):
        _, loss_val = sess.run([train, loss], feed_dict=feed_data)
        # print(f'step = {step + 1}, loss = {loss_val}, a = {x_val}, b = {y_val}')
        print(f'step = {step + 1}, loss = {loss_val}')

    # model 최적화
    a_up, b_up = sess.run([a, b])
    print(f'수정된 기울기 : {a_up}, 절편 : {b_up}')

    feed_data_test = {X: x_test, Y: y_test}  # 테스트용 공급 데이터

    # Y(정답) vs model(예측치)
    y_true = sess.run(Y, feed_dict=feed_data_test)
    y_pred = sess.run(model, feed_dict=feed_data_test)
    mse = mean_squared_error(y_true, y_pred)
    print(mse)

'''
1트 : 학습율 = 0.5, 반복 100회, MSE = 0.48196945
2트 : 학습율 = 0.4, 반복 100회, MSE = 0.58701740
3트 : 학습율 = 0.4, 반복 200회, MSE = 0.61567754
4트 : 학습율 = 0.1, 반복 100회, MSE = 0.98653430
5트 : 학습율 = 0.8, 반복 100회, MSE = 0.47670314
'''
