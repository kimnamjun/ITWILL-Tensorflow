"""
Tensorflow2.0 Keras + MNIST(0~9) + Flatten layer + History

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
import matplotlib.pyplot as plt

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
model_fit = model.fit(x=x_train, y=y_train, epochs=15, verbose=1, validation_data=(x_validation, y_validation))

# 6. model history
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

train_loss = model_fit.history['loss']
train_acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss']
val_acc = model_fit.history['val_accuracy']

# train vs val loss
plt.plot(train_loss, color='y', label='train loss')
plt.plot(val_loss, color='r', label='val loss')
plt.legend(loc='best')
plt.xlabel('epochs')
plt.show()

# train vs val accuracy
plt.plot(train_acc, color='y', label='train acc')
plt.plot(val_acc, color='r', label='val acc')
plt.legend(loc='best')
plt.xlabel('epochs')
plt.show()