"""
python code vs tensorflow code
"""

# python : 직접 실행 환경
x = 10
y = 20
z = x + y
print(z)

# import tensorflow as tf : ver 2.0
import tensorflow.compat.v1 as tf  # ver 1.x : migration
tf.disable_v2_behavior()  # ver 2.x 사용 안 함
print(tf.__version__)

''' 프로그램 정의 영역 '''
x = tf.constant(10)  # 상수 정의
y = tf.constant(20)
print(x, y)

z = x + y  # 식 정의
print(z)  # Tensor("add:0", shape=(), dtype=int32)

# session 객체 생성
sess = tf.Session()  # 상수, 변수, 식 -> device(CPU, GPU, TPU) 할당

''' 프로그램 실행 영역 '''
print('x =', sess.run(x))
print('y =', sess.run(y))
# sess(x, y) : error

x_val, y_val = sess.run([x, y])

print('z =', sess.run(z)) # x, y 상수 참조 -> 연산

# 객체 닫기
sess.close()