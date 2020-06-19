'''
문) wine data set을 이용하여 다음과 같이 DNN 모델을 생성하시오.
  <조건1>   
   - Hidden layer : relu()함수 이용  
   - Output layer : softmax()함수 이용 
   - 2개의 은닉층을 갖는 DNN 분류기
     hidden1 : nodes = 6
     hidden2 : nodes = 3  
  <조건2> hyper parameters
    learning_rate = 0.01
    iter_size = 1,000
  <조건3>  
    train/test(80:20)
    x_data : 정규화 
    y_data : one-hot encoding
'''
import numpy as np
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.datasets import load_wine # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score

# 1. wine data load
wine = load_wine()

# 2. 변수 선택/전처리  
x_data = wine.data # 178x13
y_data = wine.target # 3개 domain
print(y_data) # 0-2
print(x_data.shape) # (178, 13)

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding : 0=[1,0,0] / 1=[0,1,0] / 2=[0,0,1]
num_class = np.max(y_data)+1 # 2+1
print(num_class) # 3

y_data = np.eye(num_class)[y_data]
print(y_data.shape) # (178, 3)

# 4. train/test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=123)

# 5. X,Y 변수 정의
inputs = 13
hidden_node1 = 6
hidden_node2 = 3
outputs = 3

X = tf.placeholder(tf.float32, shape=[None, inputs]) # [n, 13개 원소]
Y = tf.placeholder(tf.float32, shape=[None, outputs]) # [n, 3개 원소]

# 6. Hypter parameters
learning_rate = 0.01
iter_size = 1000

w1 = tf.Variable(tf.random_normal([inputs, hidden_node1]))
b1 = tf.Variable(tf.random_normal([hidden_node1]))
hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]))
b2 = tf.Variable(tf.random_normal([hidden_node2]))
hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

w3 = tf.Variable(tf.random_normal([hidden_node2, outputs]))
b3 = tf.Variable(tf.random_normal([outputs]))

# 4. softmax 분류기
model = tf.matmul(hidden_output2, w3) + b3
softmax = tf.nn.softmax(model)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

y_pred = tf.argmax(softmax, axis=1)
y_true = tf.argmax(Y, axis=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_data_train = {X: x_train, Y: y_train}
    feed_data_test = {X: x_test, Y: y_test}

    for step in range(iter_size):
        _, loss_val = sess.run([train, loss], feed_data_train)
        if (step + 1) % 50 == 0:
            print(f'step = {step + 1}, loss = {loss_val}')

    y_pred_re = sess.run(y_pred, feed_data_test)
    y_true_re = sess.run(y_true, feed_data_test)

    print(f'y pred = {y_pred_re}')
    print(f'y true = {y_true_re}')
    acc = accuracy_score(y_true_re, y_pred_re)
    print(acc)
