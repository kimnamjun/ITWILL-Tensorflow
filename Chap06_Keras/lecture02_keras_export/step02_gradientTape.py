"""
자동 미분계수
- tf.GradientTape() 클래스 이용
- 역방향 step 이용 (df : 순방향 : 연산과정 -> loss)
- 딥러닝 모델 최적화 핵심 기술
- 가중치(w)에 대한 오차(loss)의 미분값을 계산
    -> x(w)에 대한 y(loss)의 기울기 계산
"""
import tensorflow as tf

"""
한 점 A(2, 3)를 지나는 접선의 기울기
2차 방정식 = y = x^2 + x
"""
with tf.GradientTape() as tape:
    x = tf.Variable(2.0)
    y = tf.math.pow(x, 2) + x

    grad = tape.gradient(y, x)  # x에 대한 y의 기울기
    print(f'기울기 = {grad.numpy()}')

# [실습] x=2.0 -> x=1.0
with tf.GradientTape() as tape:
    x = tf.Variable(1.0)
    y = tf.math.pow(x, 2) + x

    grad = tape.gradient(y, x)  # x에 대한 y의 기울기
    print(f'기울기 = {grad.numpy()}')

