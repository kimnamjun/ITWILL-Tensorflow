"""
- Tensorflow 2.0 특징
1. 즉시 실행(eager execution) 모드
    - session object 없이 즉시 실행하는 환경 (auto graph)
    - python 실행 환경과 동일함
    - API 정리 : tf.global_variables_initializer 삭제
"""
import tensorflow as tf
print(tf.__version__)

# 상수 정의
a = tf.constant([[1, 2, 3], [1.0, 2.5, 3.5]])  # [2, 3]
print(a)
print(a.numpy())

# 식 정의 : 상수 참조 -> 즉시 연산
b = tf.add(a, 0.5)
print(b)

# 변수 정의
x = tf.Variable([10,20,30])
y = tf.Variable([1,2,3])
print(x.numpy())
print(y.numpy())

mul = tf.multiply(x, y)
print('multiply', mul.numpy())

# python code -> tensorflow 즉시 실행
x = [[2.0, 3.0]]
a = [[1.0], [1.5]]

# 행렬곱 연산
mat = tf.matmul(x, a)
print(mat)

print('--------------------------------------------------------------------------------')

# 상수 정의
x = tf.constant([1.5, 2.5, 3.5], name='x')  # 1차원 : 수정 불가
print('x = ', x)  # x =  Tensor("x:0", shape=(3,), dtype=float32)

# 변수 정의
y = tf.Variable([1.0, 2.0, 3.0], name='y')  # 1차원 : 수정 가능
print('y = ', y)  # y =  <tf.Variable 'y:0' shape=(3,) dtype=float32_ref>

# 식 정의
mul = x * y

print('x = ', x.numpy())
print('y = ', y.numpy())
print('mul = ', mul.numpy())
