"""
tf.GradientTape + regression model
-> 미분계수 자동 계산 -> model 최적화(최적의 기울기와 절편 update)
"""
import tensorflow as tf
tf.executing_eagerly()  # 즉시 실행 test : ver2.x 사용중

# 1. input/output 변수 정의
inputs = tf.Variable([1.0, 2.0, 3.0])  # x변수
outputs = tf.Variable([2.0, 4.0, 6.0])  # y변수

# 2. model : Model 클래스
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()  # 부모클래스 생성자 호출
        self.W = tf.Variable(tf.random.normal([1]))  # 기울기(가중치)
        self.B = tf.Variable(tf.random.normal([1]))  # 절편

    def call(self, inputs):  # 메소드 재정의
        return inputs * self.W + self.B  # 회귀방정식(예측치)


# 2. 손실 함수 : 오차 반환
def loss(model, inputs, outputs):
    err = model(inputs) - outputs
    return tf.reduce_mean(tf.square(err))  # MSE


# 3. 미분계수(기울기) 계산
def gradient(model, inputs, outputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs)  # 손실함수 호출
        grad = tape.gradient(loss_value, [model.W, model.B])
        # 미분계수 -> 기울기와 절편 업데이트
    return grad  # 업데이트 결과 반환


# 4. model 생성
model = Model() # 생성자
mse = loss(model, inputs, outputs)
print(f'MSE = {mse.numpy()}')

grad = gradient(model, inputs, outputs)
print(f'Gradient = {grad}')

# 5. model 최적화 객체
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

print(f"초기 손실값 : {format(loss(model, inputs, outputs), '.6f')}")
print(f"w: {model.W.numpy()}, b: {model.B.numpy()}")

# 6. 반복학습
for step in range(300):
    grad = gradient(model, inputs, outputs)  # 기울기 계산
    # 기울기 -> 최적화 객체 반영
    opt.apply_gradients(zip(grad, [model.W, model.B]))

    if step % 30 == 29:
        print(f"step {step + 1}, loss = {format(loss(model, inputs, outputs), '.6f')}")

# model 최적화
print(f"최종 손실값 : {format(loss(model, inputs, outputs), '.6f')}")
print(f"w: {model.W.numpy()}, b: {model.B.numpy()}")

# model test
y_pred = model.call(2.5)
print(f'y pred = {y_pred.numpy()}')