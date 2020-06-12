"""
난수 상수 생성 함수 : 정규분포난수, 균등분포난수
tf.Variable(난수 상수) -> 변수 값 수정
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 난수
num = tf.constant(10.0)

# 0차원(Scalar) 변수
var = tf.Variable(num + 20.0)
print("var = ", var)  # var =  <tf.Variable 'Variable:0' shape=() dtype=float32_ref>

# 1차원 변수
var1d = tf.Variable(tf.random_normal([3]))  # 1차원 : [n]

# 2차원 변수
var2d = tf.Variable(tf.random_uniform([3, 2]))  # 2차원 : [r, c]

# 3차원 변수
var3d = tf.Variable(tf.random_normal([3, 2, 4]))  # 3차원 : [s, r, c]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 변수 초기화(초기값 할당) : var, var1d, var2d

    print('var =', sess.run(var))
    print('var1d =', sess.run(var1d))
    print('var2d =', sess.run(var2d))
    print('var3d =', sess.run(var3d))

    # 변수의 값 수정
    var1d_data = [0.1, 0.2, 0.3]
    print('var1d assign add =', sess.run(var1d.assign_add(var1d_data)))
    print('var1d assign =', sess.run(var1d.assign(var1d_data)))

    var3d_re = sess.run(var3d)  # numpy 객체

    print(var3d_re[0])  # 첫번째 면
    print(var3d_re[0, 0])  # 1면 1행

    # 24개 균등분포난수를 생성하여 var3d 변수에 값을 수정하시오.
    rand = tf.random_uniform([24])
    reshape = tf.reshape(rand, [3, 2, 4])  # reshape 사용 예시
    print('answer =', sess.run(var3d.assign(reshape)))