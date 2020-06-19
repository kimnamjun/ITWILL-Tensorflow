"""
1. Tfidf 가중치 기법 : sparse matrix
2. Sigmoid 활성 함수 : ham(0) / spam(1)
3. hyper parameters
    max features = 4000(input node)
    lr = 0.01
    epochs = 50
    batch size = 500
    iter size = 10
        -> 1 epoch = 500 * 10 = 5000
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# step04 실행하면 파일 있음
x_train, x_test, y_train, y_test = np.load('C:/ITWILL/6_Tensorflow/data/spam_data.npy', allow_pickle=True)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

max_features = 4000
lr = 0.01
epochs = 50
batch_size = 500
iter_size = 10

X = tf.placeholder(dtype=tf.float32, shape=[None, max_features])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

hidden_node1 = 6
hidden_node2 = 3

w1 = tf.Variable(tf.random_normal([max_features, hidden_node1]))
b1 = tf.Variable(tf.random_normal([hidden_node1]))
hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]))
b2 = tf.Variable(tf.random_normal([hidden_node2]))
hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

w3 = tf.Variable(tf.random_normal([hidden_node2, 1]))
b3 = tf.Variable(tf.random_normal([1]))
model = tf.sigmoid(tf.matmul(hidden_output2, w3) + b3)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=model))
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

y_pred = tf.cast(model > 0.5, tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        total_loss = 0

        for step in range(iter_size):
            idx = np.random.choice(a=x_train.shape[0], size=batch_size, replace=False)
            _, loss_val = sess.run([train, loss], {X: x_train[idx], Y: y_train[idx]})
            total_loss += loss_val
        avg_loss = total_loss / iter_size
        print(f'epoch: {epoch + 1}, loss: {avg_loss}')

    y_true_result = sess.run(Y, {X: x_test, Y: y_test})  # 굳이 안넣어도 됨
    y_pred_result = sess.run(y_pred, {X: x_test, Y: y_test})

    acc = accuracy_score(y_true_result, y_pred_result)
    print(f'accuracy: {acc}')
