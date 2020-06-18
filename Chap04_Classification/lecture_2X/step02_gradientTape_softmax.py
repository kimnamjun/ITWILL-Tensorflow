# -*- coding: utf-8 -*-
"""
step02_gradientTape_softmax.py

 - GradientTape + Softmax 
"""

import tensorflow as tf # ver2.x
tf.executing_eagerly() # True

# 1. input/output 변수 정의 
# [털, 날개]
inputs = tf.Variable(
    [[0., 0.], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]]) # [6, 2]

# [기타, 포유류, 조류] : [6, 3] -> one hot encoding
outputs = tf.Variable([
    [1., 0., 0.],  # 기타[0]
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])


# 2. model : Model 클래스 
class Model(tf.keras.Model) : # 자식클래스(부모클래스)
    def __init__(self) : # 생성자 
        super().__init__() # 부모생성자 호출 
        self.W = tf.Variable(tf.random.normal([2, 3])) # 기울기[입력,출력]
        self.B = tf.Variable(tf.random.normal([3])) # 절편[출력]
    def call(self, inputs) : # 메서드 재정의 
        return tf.matmul(inputs, self.W) + self.B # 회귀방정식(예측치) 

# 3. 손실 함수 : 오차 반환 
def loss(model, inputs, outputs):
    softmax = tf.nn.softmax(model(inputs)) # 활성함수 변경  
    return -tf.reduce_mean(outputs * tf.math.log(softmax) + (1-outputs) * tf.math.log(1-softmax)) # Cross Entropy 

# 4. 미분계수(기울기) 계산 
def gardient(model, inputs, outputs) :  
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs) # 손실함수 호출 
        grad = tape.gradient(loss_value, [model.W, model.B])
    return grad # 업데이터 결과 반환 

# 5. model 생성 
model = Model() # 생성자    

# 6. model 최적화 객체 
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

print("최초 손실값  : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# 7. 반복학습 
for step in range(300) :
    grad = gardient(model, inputs, outputs) #  기울기 계산 
    # 기울기 -> 최적화 객체 반영 
    opt.apply_gradients(zip(grad, [model.W, model.B]))
    
    if (step+1) % 20 == 0 :
        print("step = {}, loss = {:.6f}".format((step+1), 
                                            loss(model, inputs, outputs)))
    
# model 최적화 
print("최종 손실값  : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))
    
# model test 
y_true = tf.argmax(outputs, axis=1) # 정답 -> decoding
y_pred = tf.argmax(tf.nn.softmax(model.call(inputs)), axis=1) # 확률값 -> decoding

print("정답 : ", y_true.numpy())
print("예측치 : ", y_pred.numpy())
'''
정답 :  [0 1 2 0 0 2]
예측치 :  [0 1 2 0 0 2]
'''













