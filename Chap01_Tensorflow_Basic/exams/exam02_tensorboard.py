'''
문) 다음과 같은 상수와 사칙연산 함수를 이용하여 data flow의 graph를 작성하여 
    tensorboard로 출력하시오.
    조건1> 상수 : x = 100, y = 50
    조건2> 계산식 : result = ((x - 5) * y) / (y + 20)
       -> 사칙연산 함수 이용 계산식 작성  
        1. sub = (x - 5) : tf.subtract(x, 5)
        2. mul = ((x - 5) * y) : tf.multiply(sub, y)
        3. add = (y + 20) : tf.add(y, 20)
        4. div = mul / add : tf.div(mul, add)
'''
import tensorflow.compat.v1 as tf  # ver 1.x

tf.disable_v2_behavior()  # ver 2.x 사용안함

tf.reset_default_graph()

# 상수 정의 
x = tf.constant(100, name='x')
y = tf.constant(50, name='y')

# 계산식 
# result = ((x - 5) * y) / (y + 20)
a = tf.subtract(x, 5, name='sub')
b = tf.multiply(a, y, name='mul')
c = tf.add(y, 20, name='add')
d = tf.div(b, c, name='div')

with tf.Session() as sess:
    tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/ITWILL/6_Tensorflow/graph', sess.graph)
    writer.close()