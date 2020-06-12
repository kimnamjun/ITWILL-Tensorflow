"""
- Tensorflow 2.0 특징
3. @tf.function 함수 장식자(데코레이터)
    - 함수 장식자 이점:
        -> python code -> tensorflow code 변환(auto graph)
        -> logic 처리 : 쉬운 코드로 대체
        -> 속도 향상
"""
import tensorflow as tf
''' chap01 step07 -> ver 2.0'''
'''
# 1. if
def true_fn():
    return tf.multiply(x, 10)

def false_fn():
    return tf.add(x, 10)

x = tf.constant(10)
y = tf.cond(x > 100, true_fn, false_fn)

# 2. while
i = tf.constant(0)  # i = 0 : 반복변수

def cond(i):
    return tf.less(i, 100)  # i < 100

def body(i):
    return tf.add(i, 1)  # i += 1

loop = tf.while_loop(cond=cond, body=body, loop_vars=(i,))

sess = tf.Session()
print('y =', sess.run(y))
print('loop =', sess.run(loop))
'''
@tf.function
def if_func(x):
    # python code -> tensorflow code
    if x > 100:
        y = x * 10
    else:
        y = x + 10
    return y

x = tf.constant(10)
print(if_func(x).numpy())

@tf.function
def while_func(i):
    while i < 100:
        i += 1
    return i

i = tf.constant(0)
print(while_func(i).numpy())