'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 ANN 모델을 생성하시오.
  <조건1>   
   - 1개의 은닉층을 갖는 ANN 분류기
   - hidden nodes = 4
   - Hidden layer : relu()함수 이용  
   - Output layer : sigmoid()함수 이용 
     
  <조건2> hyper parameters
    최적화 알고리즘 : AdamOptimizer
    learning_rate = 0.0 ~ 0.01
    반복학습 : 300 ~ 500
'''

import tensorflow.compat.v1 as tf  # ver1.x

tf.disable_v2_behavior()  # ver2.0 사용안함
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import pandas as pd

bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# label에서 normal, fat 추출 
bmi = bmi[bmi.label.isin(['normal', 'fat'])]
print(bmi.head())

# 칼럼 추출 
col = list(bmi.columns)
print(col)

# x,y 변수 추출 
x_data = bmi[col[:2]]  # x변수
y_data = bmi[col[2]]  # y변수

# y변수(label) 로짓 변환 dict
map_data = {'normal': 0, 'fat': 1}
y_data = y_data.map(map_data)  # dict mapping


# x_data 정규화 함수 
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


x_data = x_data.apply(normalize)

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.transpose(np.array([y_data]))  # (1, 15102) -> (15102, 1)

print(x_data.shape)  # (15102, 2)
print(y_data.shape)  # (15102, 1)

# x,y 변수 정의
X = tf.placeholder(tf.float32, shape=[None, 2])  # x 데이터 수
Y = tf.placeholder(tf.float32, shape=[None, 1])  # y 데이터 수

tf.set_random_seed(1234)

hidden_node = 4

w1 = tf.Variable(tf.random_normal([2, hidden_node]))
b1 = tf.Variable(tf.random_normal([hidden_node]))

w2 = tf.Variable(tf.random_normal([hidden_node, 1]))
b2 = tf.Variable(tf.random_normal([1]))

hidden_output = tf.nn.relu(tf.matmul(X, w1) + b1)

model = tf.matmul(hidden_output, w2) + b2
sigmoid = tf.sigmoid(model)

loss = -tf.reduce_mean(Y * tf.log(sigmoid) + (1 - Y) * tf.log(1 - sigmoid))

train = tf.train.AdamOptimizer(0.01).minimize(loss)  # 오차 최소화

cut_off = tf.cast(sigmoid > 0.5, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_data = {X: x_data, Y: y_data}

    for step in range(500):
        _, loss_val = sess.run([train, loss], feed_data)
        if (step + 1) % 50 == 0:
            print(f'step = {step + 1}, loss = {loss_val}')

    y_true = sess.run(Y, feed_data)
    y_pred = sess.run(cut_off, feed_data)

    print(accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred))
