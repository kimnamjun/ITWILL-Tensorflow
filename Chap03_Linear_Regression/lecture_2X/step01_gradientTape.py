"""
1. 미분(gradient)계수 자동 계산
- u자 곡선에서 접선의 기울기
- tf.GradientTape : 미분계수 자동 계산 클래스

2. 저수준 API vs 고수준 API
- 저수준(low level) API : 모델, 레이어 직접 어려움 -> 코드 작성 어려움(원리 이해)
- 고수준(high level) API : 모델, 레이어 작성 쉬움 -> 코드 작성 쉬움
"""

import tensorflow as tf
tf.executing_eagerly() # 즉시 실행 test : ver2.x 사용중 

######################################
# 미분(gradient)계수 자동 계산
######################################

# 1. input/output 변수 정의 
inputs = tf.Variable([1.0]) # x변수 
outputs = tf.Variable([1.25]) # y변수
print("outputs =", outputs.numpy())   

# 2. model : 예측치
def model(inputs) :    
    y_pred = inputs * 1.0 + 0.5 # 회귀방정식    
    print("y_pred =", y_pred.numpy()) 
    return y_pred  
 
# 2. 손실 함수
def loss(model, inputs, outputs):
  err = model(inputs) - outputs
  return tf.reduce_mean(tf.square(err)) # MSE

# 3. 미분계수(기울기) 계산  
with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, outputs) # 손실함수 호출  
    print("손실값 =", loss_value.numpy())
    grad = tape.gradient(loss_value, inputs, outputs) 
    print("미분계수 =", grad.numpy())  

'''
outputs < y_pred : 미분계수 > 0
outputs > y_pred : 미분계수 < 0
'''