"""
단순 선형 회귀 방정식 : X(1) -> Y
    - y_pred = X * a(기울기) + b(절편)
    - err = Y - y_pred
    - loss function(cost function) : 정답과 예측치 간의 오차 반환 함수
    -> functiohn(Y, y_pred) -> 오차(손실 or 비용) 반환 : MSE
"""
import tensorflow as tf

# X, Y 변수 정의 : 수정 불가
X = tf.constant(6.5)
Y = tf.constant(5.2)

# a, b 변수 정의 : 수정 가능
a = tf.Variable(0.5)
b = tf.Variable(1.5)

# 회귀모델 함수
def linear_model(X):  # X : 입력
    y_pred = tf.math.add(tf.math.multiply(X, a), b)
    return y_pred

# 모델 오차
def model_err(X, Y):  # (입력, 정답)
    y_pred = linear_model(X)
    err = tf.math.subtract(Y, y_pred)
    return err

# 손실함수(loss function) : (정답, 예측치) -> 오차 반환(MSE)
def loss_function(X, Y):
    err = model_err(X, Y)
    loss = tf.reduce_mean(tf.square(err))  # MSE
    return loss

'''
오차 : MSE
error : 정답 - 예측치
square : 부호(+), 패널티
'''

print('최초 기울기(a)와 절편(b)')
print('a = {}, b = {}'.format(a.numpy(), b.numpy()))
print('model error =', model_err(X, Y).numpy())
print('loss function =', loss_function(X, Y).numpy())

# 2차식 : a = 0.6, b = 1.2 (기울기, 절편 수정)
a.assign(0.6)  # 기울기 수정(0.5 -> 0.6)
b.assign(1.2)  # 절편 수정(1.5 -> 1.2)
print('\n2차 기울기와 절편 수정')
print('a = {}, b = {}'.format(a.numpy(), b.numpy()))
print('model error =', model_err(X, Y).numpy())
print('loss function =', loss_function(X, Y).numpy())


'''
[키워드 정리]
최적화된 모델 : 최적의 기울기와 절편 수정 -> 손실(loss) 0에 수렴
딥러닝 최적화 알고리즘 : GD, Adam -> 최적의 기울기와 절편 수정 역할
'''