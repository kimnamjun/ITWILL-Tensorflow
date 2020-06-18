"""
softmax + MNIST
"""
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

# 2. images 전처리
# 1) 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2) 3차원 -> 2차원
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 3. labels 전처리
# 1) 1차원 -> 2차원
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 2) one-hot encoding
one_hot_encoder = OneHotEncoder()
y_train = one_hot_encoder.fit_transform(y_train).toarray()
y_test = one_hot_encoder.fit_transform(y_test).toarray()

# 4. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

# 5. softmax 알고리즘
# 1) model
model = tf.matmul(X, w) + b

# 2) softmax
softmax = tf.nn.softmax(model)  # 활성함수

# 3) loss function : Softmax + Cross Entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# 4) optimizer
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# 5) encoding -> decoding
y_true = tf.argmax(softmax, axis=1)
y_pred = tf.argmax(softmax, axis=1)

# 6. model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_data_train = {X: x_train, Y: y_train}
    feed_data_test = {X: x_test, Y: y_test}

    # 반복학습 : 300회
    for step in range(300):
        _, loss_val = sess.run([train, loss], feed_data_train)
        print(f'step: {step+1}, loss: {loss_val}')

    # model test
    y_true_result = sess.run(y_true, feed_data_test)
    y_pred_result = sess.run(y_pred, feed_data_test)

    acc = accuracy_score(y_true_result, y_pred_result)
    print(f'accuracy: {acc}')