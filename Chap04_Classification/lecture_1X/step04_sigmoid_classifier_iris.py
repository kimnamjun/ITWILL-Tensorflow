"""
- 활성 함수(activation function)
- 손실 함수(loss function)
"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. x, y 공급 data
iris = load_iris()
x_data = iris.data[:100]
y_data = iris.target[:100].reshape(-1, 1)


# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])  # [관측치, 입력수]
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # [관측치, 출력수]

# 3. w, b 변수 정의
w = tf.Variable(tf.random_normal([4, 1]))  # [입력수]
b = tf.Variable(tf.random_normal([1]))  # [출력수]

# 4. sigmoid 분류기
# (1) model : 회귀방정식
model = tf.matmul(X, w) + b
sigmoid = tf.sigmoid(model)  # 활성함수 적용 (0~1)

# (2) loss function : Entropy 이용 = -sum(Y * log(model))
loss = -tf.reduce_mean(Y * tf.log(sigmoid) + (1-Y) * tf.log(1-sigmoid))

# (3) optimizer
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 오차 최소화

# (4) cut-off : 0.5
cut_off = tf.cast(sigmoid > 0.5, dtype=tf.float32)

# 5. model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # w, b 초기화
    feed_data = {X: x_data, Y: y_data}  # 공급 데이터

    # 반복학습 : 500회
    for step in range(500):
        _, loss_val = sess.run([train, loss], feed_data)

        if (step+1) % 50 == 0:
            print(f'step = {step+1}, loss = {loss_val}')

    # model 최적화
    y_true = sess.run(Y, feed_data)
    y_pred = sess.run(cut_off, feed_data)

    acc = accuracy_score(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))
    print(acc)
