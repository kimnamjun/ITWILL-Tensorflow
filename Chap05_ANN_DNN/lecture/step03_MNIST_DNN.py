"""
DNN model + MNIST + hyper parameter + mini batch
- Network layer
- input nodes : 28 x 28 = 784
- hidden node1 : 128 - 1층
- hidden node2 : 64 - 2층
- output node : 10 - 3층

- hyper parameter
- lr(learning rate) : 학습율
- epoch : 전체 dataset 재사용 횟수
- batch size : 1회 data 공급 횟수
- iter size : 반복 횟수
    -> 1 epoch(60000) : batch size(200) * iter size(300)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. MNIST dataset load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # images(픽셀) : (60000, 28, 28)
print(y_train.shape)  # labels(10진수) : (60000,)

###############
# DNN network #
###############
lr = 0.01
epochs = 20
batch_size = 200
iter_size = y_train.shape[0] // batch_size

inputs = 784
hidden_node1 = 128
hidden_node2 = 64
outputs = 10

# 2. images 전처리
# 1) 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2) 3차원 -> 2차원
x_train = x_train.reshape(-1, inputs)
x_test = x_test.reshape(-1, inputs)

# 3. labels 전처리
# 1) 1차원 -> 2차원
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 2) one-hot encoding
one_hot_encoder = OneHotEncoder()
y_train = one_hot_encoder.fit_transform(y_train).toarray()
y_test = one_hot_encoder.fit_transform(y_test).toarray()

# 4. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None, outputs])


# 5. softmax 알고리즘
# 1) model
w1 = tf.Variable(tf.random_normal([inputs, hidden_node1]))
b1 = tf.Variable(tf.random_normal([hidden_node1]))
hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]))
b2 = tf.Variable(tf.random_normal([hidden_node2]))
hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

w3 = tf.Variable(tf.random_normal([hidden_node2, outputs]))
b3 = tf.Variable(tf.random_normal([outputs]))
model = tf.matmul(hidden_output2, w3) + b3

# 2) softmax
softmax = tf.nn.softmax(model)  # 활성함수

# 3) loss function : Softmax + Cross Entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# 4) optimizer
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 5) encoding -> decoding
y_true = tf.argmax(softmax, axis=1)
y_pred = tf.argmax(softmax, axis=1)

# 6. model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 반복학습 : 300회
    for epoch in range(epochs):
        total_loss = 0

        for step in range(iter_size):
            idx = np.random.choice(a=y_train.shape[0], size=batch_size, replace=False)  # a=60000, 비복원 추출
            _, loss_val = sess.run([train, loss], {X: x_train[idx], Y: y_train[idx]})
            total_loss += loss_val

        # 1 epoch 종료
        avg_loss = total_loss / iter_size
        print(f'epoch: {epoch + 1}, loss: {avg_loss}')

    # model test
    y_true_result = sess.run(y_true, {X: x_test, Y: y_test})
    y_pred_result = sess.run(y_pred, {X: x_test, Y: y_test})

    acc = accuracy_score(y_true_result, y_pred_result)
    print(f'accuracy: {acc}')
