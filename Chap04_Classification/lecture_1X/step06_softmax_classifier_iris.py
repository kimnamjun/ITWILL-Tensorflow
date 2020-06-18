"""
- 활성함수 : Softmax(model)
- 손실함수 : Cross Entropy
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder  # one hot encoding
from sklearn.metrics import accuracy_score, f1_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. x, y 공급 data
iris = load_iris()
x_data = iris.data
y_data = iris.target.reshape(-1, 1)

'''
0 -> 1 0 0
1 -> 0 1 0
2 -> 0 0 1
'''
one_hot_encoder = OneHotEncoder()
y_data =  one_hot_encoder.fit_transform(y_data).toarray()

# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])  # one hot 방식이라 출력수 3개

# 3. w, b 변수 정의 : 초기값 난수 이용
w = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

# 4. softmax 분류기
# 1) 회귀방정식 : 예측치
model = tf.matmul(X, w) + b # 회귀모델

# (2) loss function : Entropy 이용 : -sum(Y * log(model))
softmax = tf.nn.softmax(model) # softmax(예측치)

loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))                # 1차 방법
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))  # 2차 방법 (음수 부호 없어짐)

# 3) optimizer : 오차 최소화(w, b update)
train = tf.train.AdamOptimizer(0.01).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2진수) -> decoding(10)
y_pred = tf.argmax(softmax, axis=1)
y_true = tf.argmax(Y, axis=1)

# 5. model 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_data = {X: x_data, Y: y_data}

    # 반복학습 : 500회
    for step in range(5000):
        _, loss_val = sess.run([train, loss], feed_data)
        if (step+1) % 50 == 0:
            print(f'step = {step + 1}, loss = {loss_val}')

    # model result
    y_pred_re = sess.run(y_pred, feed_data)
    y_true_re = sess.run(y_true, feed_data)

    print(f'y pred = {y_pred_re}')
    print(f'y true = {y_true_re}')
    acc = accuracy_score(y_true_re, y_pred_re)
    print(acc)
    f1 = f1_score(y_true_re, y_pred_re, average='macro')
    print(f1)
    