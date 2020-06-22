"""
Tensorflow2.0 Keras + MNIST(0~9) + Flatten layer

1차 : 1차원 : (28x28) -> 784
2차 : 2차원 : 28x28 -> Flatten 적용
"""
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # Y 변수 전처리
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets.mnist import load_data

# 1. X, Y 공급 data
(x_train, y_train), (x_validation, y_validation) = load_data()

# X 변수 전처리 : 정규화, 차원 축소
x_train = x_train / 255.
# x_train = x_train.reshape(-1, 784)  # Flatten layer 있어서 생략

x_validation = x_validation / 255.
# x_validation = x_validation.reshape(-1, 784)

# Y 변수 전처리 : one hot encoding
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)

# 2. Keras Model 생성
model = Sequential()

# 3. model layer
input_shape = (28, 28)  # 2차원

# Flatten layer : 2D(28x28) -> 1D(784)
model.add(Flatten(input_shape=input_shape))  # 0층

model.add(Dense(128, activation='relu'))  # hidden layer = [784, 128]
model.add(Dense(64, activation='relu'))  # hidden layer = [128, 64], input_shape는 생략 가능인듯
model.add(Dense(32, activation='relu'))  # output layer = [64, 32]
model.add(Dense(10, activation='softmax'))  # output layer = [32, 10]


# 4. model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5. model training
model.fit(x=x_train, y=y_train, epochs=10, verbose=1, validation_data=(x_validation, y_validation))

# 6. model evaluation
model.evaluate(x=x_validation, y=y_validation)

# 7. model save / load
model.save("keras_model_mnist.h5")
print("save model")

new_model = load_model("keras_model_mnist.h5")
print("load model")
new_model.summary()

# 8. model test : new data set
y_pred = new_model.predict(x_validation)
y_true = y_validation

y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.argmax(y_true, axis=1)

print(accuracy_score(y_true, y_pred))
