"""
tf.GradientTape + regression model(iris)
- x변수 : 2~4 컬럼
- y변수 : 1컬럼
- model 최적화 알고리즘 : Adam
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
tf.executing_eagerly()

# 1. input/output 변수 정의
iris = load_iris()
inputs = iris.data[:, 1:]
outputs = iris.data[:, 0]

tf.random.set_seed(123)  # W, B seed값
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.3, random_state=123)


# 2. model : Model 클래스
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.random.normal([3, 1]))
        self.B = tf.Variable(tf.random.normal([1]))

    def call(self, inputs):  # 메소드 재정의
        return tf.matmul(tf.cast(inputs, tf.float32), self.W) + self.B


# 2. 손실 함수 : 오차 반환
def loss(model, inputs, outputs):
    err = model(inputs) - outputs
    return tf.reduce_mean(tf.square(err))


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
opt = tf.keras.optimizers.SGD(learning_rate=0.001)

print(f"초기 손실값 : {format(loss(model, x_train, y_train), '.6f')}")
print(f"w: {model.W.numpy()}, b: {model.B.numpy()}")

# 6. 반복학습
for step in range(100):
    grad = gradient(model, x_train, y_train)  # 기울기 계산
    # 기울기 -> 최적화 객체 반영
    opt.apply_gradients(zip(grad, [model.W, model.B]))

    if step % 30 == 29:
        print(f"step {step + 1}, loss = {format(loss(model, x_train, y_train), '.6f')}")

# model 최적화
print(f"최종 손실값 : {format(loss(model, x_train, y_train), '.6f')}")
print(f"w: {model.W.numpy()}, b: {model.B.numpy()}")

# model test
y_pred = model.call(x_test)
print(y_pred.numpy())
mse = mean_squared_error(y_test, y_pred)
print(f'mse = {mse}')
r2 = r2_score(y_test, y_pred)
print(f'r2 = {r2}')
