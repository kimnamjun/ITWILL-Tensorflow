"""
- 활성함수 : Softmax(model)
- 손실함수 : Cross Entropy
"""
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. x, y 공급 data
# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]])  # [6, 2]

# [기타, 포유류, 조류] : [6, 3] -> one hot encoding
y_data = np.array([
    [1, 0, 0],  # 기타[0]
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])  # one hot 방식이라 출력수 3개

# 3. w, b 변수 정의 : 초기값 난수 이용
w = tf.Variable(tf.random_normal([2, 3]))
b = tf.Variable(tf.random_normal([3]))

# 4. softmax 분류기
# 1) 회귀방정식 : 예측치
model = tf.matmul(X, w) + b # 회귀모델

# softmax(예측치)
softmax = tf.nn.softmax(model)

# (2) loss function : Entropy 이용 : -sum(Y * log(model))
loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 3) optimizer : 오차 최소화(w, b update)
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2진수) -> decoding(10)
y_pred = tf.argmax(softmax, axis=1)
y_true = tf.argmax(Y, axis=1)

# 5. model 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_data = {X: x_data, Y: y_data}

    # 반복학습 : 500회
    for step in range(500):
        _, loss_val = sess.run([train, loss], feed_data)
        if (step+1) % 50 == 0:
            print(f'step = {step + 1}, loss = {loss_val}')

    # model result
    print(sess.run(softmax, feed_data))
    y_pred_re = sess.run(y_pred, feed_data)
    y_true_re = sess.run(y_true, feed_data)

    print(f'y pred = {y_pred_re}')
    print(f'y true = {y_true_re}')
    acc = accuracy_score(y_true_re, y_pred_re)
    print(acc)