"""
초기값이 없는 변수 : Feed 방식
    변수 = tf.placeholder(dtype, shape)
    - dtype : 자료형(tf.int, tf.float, tf.string)
    - shape : 자료구조([n] : 1차원, [r,c] : 2차원, 생략 : 공급 data 결정)
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 변수 정의
a = tf.placeholder(dtype=tf.float32)  # shape 생략(가변형)
b = tf.placeholder(dtype=tf.float32)

c = tf.placeholder(dtype=tf.float32, shape=[5])  # 고정형 : 1d
d = tf.placeholder(dtype=tf.float32, shape=[None, 3])  # 고정형 : 2d (행 가변)

c_data = tf.random_uniform([5])  # 0 ~ 1 사이 난수
# 정의만 했을 뿐 실제 생성된 것은 아님

# 식 정의
mul = tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c * 0.5  # 1d * 0d

with tf.Session() as sess:
    # 변수 초기화 생략 (변수 없음)

    # 식 실행
    mul_re1 = sess.run(mul, feed_dict={a: 2.5, b: 3.5})  # data feed
    print('mul1 =', mul_re1)  # mul1 = 8.75
    a_data = [1.0, 2.0, 3.5]
    b_data = [0.5, 0.3, 0.4]
    feed_data = {a: a_data, b: b_data}
    mul_re2 = sess.run(mul, feed_dict=feed_data)
    print('mul2 =', mul_re2)  # mul2 = [0.5 0.6 1.4]

    # 식 실행 : 식 참조
    add_re = sess.run(add, feed_dict=feed_data)  # mul + 10
    print('add =', add_re)

    c_data_re = sess.run(c_data)  # 상수 생성
    print(sess.run(c_calc, feed_dict={c: c_data_re}))