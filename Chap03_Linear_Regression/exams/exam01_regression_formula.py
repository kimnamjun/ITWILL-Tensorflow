'''
문1) 다음과 같이 다중선형회귀방정식으로 모델의 예측치와 평균제곱오차를 계산하시오.
    조건1> X변수 : 4개, Y변수 : 1개  
    조건1> X변수 공급 데이터 : x_data = [[1.2,2.2,3.5,4.2]] - (1,4)
    조건2> Y변수[정답] : Variable()이용 표준정규분포 난수 상수 1개  
    조건3> a변수[기울기] : Variable()이용 표준정규분포 난수 상수 4개
    조건4> b변수[절편] : Variable()이용 표준정규분포 난수 상수 1개     
    조건5> model 예측치 : pred_Y = (X * a) + b 
        -> 행렬곱 함수 적용  
    조건6> model 손실함수 출력 
        -> 손실함수는 python 함수로 정의 : 함수명 -> loss_fn(err)

<< 출력 예시 >>    
a[기울기] = 
 [[-2.063591 ]
 [ 0.10648511]
 [ 0.49105361]
 [ 0.00555888]]
b[절편] = [[ 1.4659301]]
Y[정답] = [[-0.39188424]]
pred_Y[예측치] = [[ 0.96592301]]
loss function = = 1.84364 
'''

import tensorflow as tf  

# 1. 변수 정의 
X = [[1.2,2.2,3.5,4.2]] # 공급 data
Y = tf.Variable(tf.random.normal([1])) # 출력 
a = tf.Variable(tf.random.normal([4, 1])) # 기울기(4,1) 
b = tf.Variable(tf.random.normal([1])) # 절편

# 2. model 예측치/오차/손실함수 정의 

def pred(X):
    Y_pred = tf.math.add(tf.math.multiply(X, a), b)
    return Y_pred

def error(X, Y):
    Y_pred = pred(X)
    err = tf.math.subtract(Y, Y_pred)
    return err

def loss_fn(err):
    loss = tf.reduce_mean(tf.square(err))
    return loss

# 3. 결과 출력
print('a[기울기] =\n', a.numpy())
print('b[절편] =', b.numpy())
print('Y[정답] =', Y.numpy())
print('loss function =', loss_fn(error(X, Y)).numpy())
