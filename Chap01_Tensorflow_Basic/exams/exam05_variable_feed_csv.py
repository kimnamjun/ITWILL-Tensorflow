'''
문3) bmi.csv 파일을 가져와서 1,2칼럼은 x변수에 3칼럼은 y변수에 저장하여 처리하시오.
   조건1> x변수 : placeholder()이용 None행2열 배열 선언
   조건2> y변수 : placeholder()이용 1차원 배열 선언
   조건3> 칼럼 단위 평균 계산, label 빈도수 출력   
    
<<출력 결과 예시>>
키 평균 : 164.938
몸무게 평균 : 62.41

label 빈도수 :
normal    7677
fat       7425
thin      4898 
'''

import pandas as pd 

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# x,y 공급 data 
x_data = bmi[['height', 'weight']] # 복수 칼럼 선택 
y_data = bmi['label'] # 단일 칼럼 선택

X = tf.placeholder(dtype=tf.int32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.string, shape=[None])

with tf.Session() as sess:
    feed_data = {X: x_data, Y: y_data}
    X_val, Y_val = sess.run([X, Y], feed_dict=feed_data)

    # 키와 몸무게 평균
    print(type(X_val))
    print(X_val.shape)
    print('키 평균 :', X_val[:, 0].mean())
    print('몸무게 평균 :', X_val[:, 1].mean())

    # 빈도수
    print('label 빈도수 :')
    Y_ser = pd.Series(Y_val)
    print(Y_ser.value_counts())

