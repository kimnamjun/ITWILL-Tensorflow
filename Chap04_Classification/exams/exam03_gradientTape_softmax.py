# -*- coding: utf-8 -*-

'''
문) load_wine()
  조건1> train/test - 7:3비율 
  조건2> y 변수 : wine.target
  조건3> x 변수 : wine.data
  조건4> learning_rate = 0.1 ~ 0.01
  조건5> optimizer = tf.keras.optimizers.Adam
  조건6> epoch(step) = 300회
  조건7> 모델 평가 : Accuracy
  
  <출력결과 예시>
  초기 손실값 : 0.935543
------------------------------
Step = 020 -> loss = 0.242
Step = 040 -> loss = 0.126
Step = 060 -> loss = 0.091
Step = 080 -> loss = 0.073
Step = 100 -> loss = 0.061
Step = 120 -> loss = 0.053
Step = 140 -> loss = 0.046
Step = 160 -> loss = 0.041
Step = 180 -> loss = 0.036
Step = 200 -> loss = 0.033
Step = 220 -> loss = 0.030
Step = 240 -> loss = 0.027
Step = 260 -> loss = 0.025
Step = 280 -> loss = 0.023
Step = 300 -> loss = 0.021
------------------------------
최종 손실값 : 0.021365
분류정확도 : 0.9629629629629629 
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale # 정규화 
import tensorflow as tf 

# 1. data load
wine = load_wine()
print(wine) # "data", "target"

# 변수 선택  
X = wine.data  
y = wine.target


# X변수 정규화 
x_data = minmax_scale(X)

# y변수 one-hot encoding
obj = OneHotEncoder() 
y_data = obj.fit_transform(y.reshape([-1, 1])).toarray()


# train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=123)

x_train.shape # (124, 13)
y_train.shape # (124, 3)


# 2. Model 클래스 
class Model(tf.keras.Model): # keras Model class 상속 
  def __init__(self): # 생성자 
    super().__init__() 
    self.W = tf.Variable(tf.random.normal([13, 3])) 
    self.B = tf.Variable(tf.random.normal([3])) 
  def call(self, inputs): # 메서드 재정의 
    return tf.matmul(tf.cast(inputs, tf.float32), self.W) + self.B # 예측치  


# 3. 손실함수 : (예측치, 정답) -> 오차 
def loss(model, inputs, outputs):
    softmax = tf.nn.softmax(model(inputs))
    return -tf.reduce_mean(outputs * tf.math.log(softmax) + (1-outputs) * tf.math.log(1-softmax))

# 4. 기울기 계산 함수 : 오차값 -> 기울기 반환  
def gradient(model, inputs, outputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs)  # 손실함수 호출
        grad = tape.gradient(loss_value, [model.W, model.B])
        # 미분계수 -> 기울기와 절편 업데이트
    return grad  # 업데이트 결과 반환

# 5. 모델 및 최적화 객체   
model = Model() # 생성자
mse = loss(model, x_train, y_train)
grad = gradient(model, x_train, y_train)

print(f'MSE = {mse.numpy()}')
print(f'Gradient = {grad}')

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# 6. 반복 학습 : Model 객체와 손실함수 이용
for step in range(300):
    grad = gradient(model, x_train, y_train)  # 기울기 계산
    # 기울기 -> 최적화 객체 반영
    opt.apply_gradients(zip(grad, [model.W, model.B]))

    if step % 30 == 29:
        print(f"step {step + 1}, loss = {format(loss(model, x_train, y_train), '.6f')}")
        
print(f"최종 손실값 : {format(loss(model, x_train, y_train), '.6f')}")    

# 7. 모델 평가 : 분류정확도
y_true = tf.argmax(y_test, axis=1)  # 정답
y_pred = tf.argmax(tf.nn.softmax(model.call(x_test)), axis=1)  # 예측치
print('정답: ', y_true.numpy())
print('예측: ', y_pred.numpy())
print(accuracy_score(y_true, y_pred))