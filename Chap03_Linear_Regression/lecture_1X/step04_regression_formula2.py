"""
다중 선형 회귀 방정식 : 행렬곱 이용
- X(n) -> Y
- y_pred = X1 * a1 + X2 * a2 + ...
- y_pred = tf.matmul(X, a) + b
"""
import tensorflow as tf

# X, Y 변수 정의
X = [[1.0, 2.0]]
Y = 2.5

# a, b 변수 정의
a = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

# model 식 정의
y_pred = tf.math.add(tf.matmul(X, a), b)
print(y_pred)

# model error
err = Y - y_pred

# loss function : 손실 반환
loss = tf.reduce_mean(tf.square(err))
print('기울기(a)와 절편(b)')
print('a = {}, b = {}'.format(a.numpy(), b.numpy()))
print('model error =', err.numpy())
print('loss function =', loss.numpy())