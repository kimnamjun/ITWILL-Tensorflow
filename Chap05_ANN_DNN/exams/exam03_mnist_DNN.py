'''
문) DNN Layer를 적용하여 다음과 같이 DNN 모델을 생성하시오.
  조건1> hyper parameters
    learning_rate = 0.01
    training_epochs = 15
    batch_size = 100
    iter_size = 600
  조건2> DNN layer
    Layer1 =  784 x 512
    Layer2 =  512 x 256
    Layer3 =  256 x 10 
    
<<출력 예시>>
epoch = 1 cost= 55.16994312942028
epoch = 2 cost= 8.41355741351843
epoch = 3 cost= 4.618276522534089
epoch = 4 cost= 3.1665531673272898
epoch = 5 cost= 2.419369451348884
epoch = 6 cost= 1.9141375153187972
epoch = 7 cost= 1.4185047257376269
epoch = 8 cost= 1.3425187619523207
epoch = 9 cost= 1.1093137870219107
epoch = 10 cost= 0.9633290690958626
epoch = 11 cost= 0.7082862980902166
epoch = 12 cost= 0.7516645132078936
epoch = 13 cost= 0.5149420341080038
epoch = 14 cost= 0.3507416256171109
epoch = 15 cost= 0.26566603047227244
------------------------------------
accuracy = 0.9507        
'''  

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
import numpy as np

tf.set_random_seed(123) # w,b seed

# 1. MNIST dataset load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# train set shape 확인 
x_train.shape # (60000, 28, 28) -> image(입력) : (size, h, w)
y_train.shape # (60000,) -> label(정답)

# test set shape
x_test.shape # (10000, 28, 28) -> image(입력)
y_test.shape # (10000,) -> label(정답)

# 2. 전처리 : X변수 정규화, Y변수 one-hot encoding  
x_train_nor, x_test_nor = x_train / 255.0, x_test / 255.0

# one-hot encoding
obj = OneHotEncoder()
train_labels = obj.fit_transform(y_train.reshape([-1, 1])).toarray()
train_labels.shape # (60000, 10)

test_labels = obj.fit_transform(y_test.reshape([-1, 1])).toarray()
test_labels.shape # (10000, 10)


# 3. 공급 data : image reshape(3d -> 2d)
train_images = x_train_nor.reshape(60000, 784)
test_images = x_test_nor.reshape(10000, 784)
train_images.shape # (60000, 784)
test_images.shape # (10000, 784)


X = tf.placeholder(tf.float32, [None, 784]) # [관측치, 입력수]
Y = tf.placeholder(tf.float32, [None, 10]) # [관측치, 출력수]

# hyper parameters
lr = 0.01
epochs = 15 # 전체 images(60,000) 20번 재사용 
batch_size = 100 # 1회 data 공급 size
iter_size = 600 # 반복횟수 

##############################
### MNIST DNN network
##############################

inputs = 784
hidden_node1 = 512
hidden_node2 = 256
outputs = 10

x_train = x_train / 255.0
x_train = x_train.reshape(-1, inputs)
x_test = x_test / 255.0
x_test = x_test.reshape(-1, inputs)

one_hot_encoder = OneHotEncoder()

y_train = y_train.reshape(-1, 1)
y_train = one_hot_encoder.fit_transform(y_train).toarray()
y_test = y_test.reshape(-1, 1)
y_test = one_hot_encoder.fit_transform(y_test).toarray()


w1 = tf.Variable(tf.random_normal([inputs, hidden_node1]))
b1 = tf.Variable(tf.random_normal([hidden_node1]))
hidden_output1 = tf.nn.relu(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.random_normal([hidden_node1, hidden_node2]))
b2 = tf.Variable(tf.random_normal([hidden_node2]))
hidden_output2 = tf.nn.relu(tf.matmul(hidden_output1, w2) + b2)

w3 = tf.Variable(tf.random_normal([hidden_node2, outputs]))
b3 = tf.Variable(tf.random_normal([outputs]))
final_output = tf.nn.softmax(tf.matmul(hidden_output2, w3) + b3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=final_output))

train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

y_true = tf.argmax(final_output, axis=1)
y_pred = tf.argmax(final_output, axis=1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

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
