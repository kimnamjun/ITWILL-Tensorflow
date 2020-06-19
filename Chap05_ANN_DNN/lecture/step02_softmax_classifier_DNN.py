"""
DNN model
- hidden layer : relu 활성함수
- output layer : Softmax 활성함수
- 2개 은닉층을 갖는 분류기
- hidden1 nodes = 12개
- hidden2 nodes = 6개
- dataset : iris
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  # one hot encoding
from sklearn.metrics import accuracy_score, f1_score
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# 1. x, y 공급 data
iris = load_iris()
x_data = iris.data
y_data = iris.target.reshape(-1, 1)

one_hot_encoder = OneHotEncoder()
y_data = one_hot_encoder.fit_transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])  # one hot 방식이라 출력수 3개


################
# DNN network #
################
# 3. w, b 변수 정의
hidden_node1 = 12
hidden_node2 = 6

# hidden layer1 : 1층
w1 = tf.Variable(tf.random_normal([4, hidden_node1]))
b1 = tf.Variable(tf.random_normal([hidden_node1]))
hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden layer2 : 2층
w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]))
b2 = tf.Variable(tf.random_normal([hidden_node2]))
hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

# output layer : 3층
w3 = tf.Variable(tf.random_normal([hidden_node2, 3]))
b3 = tf.Variable(tf.random_normal([3]))


# 4. softmax 분류기
# 1) 회귀방정식 : 예측치
model = tf.matmul(hidden_output2, w3) + b3
softmax = tf.nn.softmax(model)

# (2) loss function : Entropy 이용 : -sum(Y * log(model))
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
        if (step + 1) % 50 == 0:
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
