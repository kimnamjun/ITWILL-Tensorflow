'''
선형대수 연산 함수
 tf.transpose : 전치행렬   
 tf.diag : 대각행렬 -> tf.linalg.diag(x)  
 tf.matrix_determinant : 정방행렬의 행렬식 -> tf.linalg.det(x)
 tf.matrix_inverse : 정방행렬의 역행렬 -> tf.linalg.inv(x)
 tf.matmul : 두 텐서의 행렬곱 -> tf.linalg.matmul(x, y)
'''

import tensorflow as tf
import numpy as np

# 정방행렬 데이터 생성 
x = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
y = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 

tran = tf.transpose(x) # 전치행렬
dia = tf.linalg.diag(x) # 대각행렬 
mat_deter = tf.linalg.det(x) # 정방행렬의 행렬식  
mat_inver = tf.linalg.inv(x) # 정방행렬의 역행렬
mat = tf.linalg.matmul(x, y) # 행렬곱 반환 

print(x)
print(tran)  
print(dia) 
print(mat_deter)
print(mat_inver)
print(mat)

# 단위행렬 -> ont-hot encoding
a = [0, 1, 2]
encoding = np.eye(len(a))[a]
print(encoding)

# tf.multiply vs tf.matmul
'''
tf.multiply : 브로드 캐스트
    - X * a -> input(1)
tf.matmul : 행렬곱
    - X1 * a1 + X2 * a2 -> input(n)
'''

'''
기상 0530
출발 0540
강남역 0740
디엠시 0900

디엠시 1800
강남역 1910
집 2100
'''


















