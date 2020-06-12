"""
Tensorflow 변수 유형
1. 초기값을 갖는 변수 : Fetch 방식
    변수 = tf.Variable(초기값)
2. 초기값이 없는 변수 : Feed 방식
    변수 = tf.placeholder(dtype, shape)
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 상수 정의
x = tf.constant(100.0)
y = tf.constant(50.0)

# 식 정의
add = tf.add(x, y)

# 변수 정의
var1 = tf.Variable(add)  # Fetch 방식 : 초기값
var2 = tf.placeholder(dtype=tf.float32)  # Feed 방식 : 초기값 X

# 변수 참조하는 식
mul1 = tf.multiply(x, var1)
mul2 = tf.multiply(x, var2)

with tf.Session() as sess:
    print('add = ', sess.run(add))  # 식 실행
    sess.run(tf.global_variables_initializer())  # 변수 초기화(Fetch 방식)
    print('var1 =', sess.run(var1))
    print('var2 =', sess.run(var2, feed_dict={var2: 150}))

    mul_re1 = sess.run(mul1)  # 상수와(100) 변수(150) 참조
    print('mul1 =', mul_re1)

    # feed 방식의 식 연산 수행
    mul_re2 = sess.run(mul2, feed_dict={var2: 150})
    print('mul2 =', mul_re2)
