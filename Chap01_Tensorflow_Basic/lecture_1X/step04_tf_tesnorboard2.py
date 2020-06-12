"""
name_scope 이용 : 영역별 tensorflow 시각화
- model 생성 -> model 오차 -> model 평가
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()

# 상수 정의 : X, a, b, Y
X = tf.constant(5.0, name='x_data')  # 입력 X
a = tf.constant(10.0, name='a')  # 기울기
b = tf.constant(4.45, name='b')  # 절편
Y = tf.constant(55.0, name='Y')  # 정답 Y

# name_scope : 회귀 방정식 정의
with tf.name_scope("regress_model") as scope:
    model = (X * a) + b

with tf.name_scope("model_error") as scope:
    model_err = tf.subtract(Y, model)

with tf.name_scope("model_evaluation") as scope:
    mse = tf.reduce_mean(tf.square(tf.subtract(Y, model)))

with tf.Session() as sess:
    tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/ITWILL/6_Tensorflow/graph', sess.graph)
    writer.close()
    print("X = ", sess.run(X))
    print("Y = ", sess.run(Y))
    print("Y pred = ", sess.run(model))
    print("model err = ", sess.run(model_err))
    print("mse = ", sess.run(mse))
