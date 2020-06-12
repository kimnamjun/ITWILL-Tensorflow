"""
- 변수 정의와 초기화
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

''' 프로그램 정의 영역 '''
# 상수 정의
x = tf.constant([1.5, 2.5, 3.5], name='x')  # 1차원 : 수정 불가
print('x = ', x)  # x =  Tensor("x:0", shape=(3,), dtype=float32)

# 변수 정의
y = tf.Variable([1.0, 2.0, 3.0], name='y')  # 1차원 : 수정 가능
print('y = ', y)  # y =  <tf.Variable 'y:0' shape=(3,) dtype=float32_ref>

# 식 정의
mul = x * y  # 상수 * 변수

sess = tf.Session()

# 변수 초기화
init = tf.global_variables_initializer()

''' 프로그램 실행 영역 '''
print('x = ', sess.run(x))  # 상수 할당 : x = [1.5, 2.5, 3.5]

sess.run(init)  # 참조 -> 변수 값 초기화
print('y = ', sess.run(y))  # 변수 할당 : y = [1. 2. 3.]

mul_re = sess.run(mul)
print('mul = ', mul_re)  # 식 할당(연산) : mul =  [ 1.5  5.  10.5]
print(type(mul_re))  # <class 'numpy.ndarray'>

print("sum =", mul_re.sum())

sess.close()