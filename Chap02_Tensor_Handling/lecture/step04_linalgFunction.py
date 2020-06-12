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
