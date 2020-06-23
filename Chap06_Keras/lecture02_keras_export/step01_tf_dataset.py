"""
Dataset 클래스
- dataset으로부터 사용가능한 데이터를 메모리에 로딩 기능
- batch size 지정 가능
"""
import tensorflow as tf
from tensorflow.python.data import Dataset

# member 확인
print(dir(Dataset))
"""
batch()
from_tensor_slices()
shuffle()
"""
# 1. from_tensor_slices() : 입력 tensor로 부터 slice 생성
# ex) MNIST(60000, 28, 28) -> 60000개 image를 각각 1개씩 slice

# 1) x, y 변수 생성
x = tf.random.normal([5, 2])
y = tf.random.normal([5])

# 2) Dataset : 5개 slice
train_ds = Dataset.from_tensor_slices((x, y))
print(train_ds)  # <DatasetV1Adapter shapes: ((2,), ()), types: (tf.float32, tf.float32)>

# 5개 관측치 -> 5개 slice
for train_x, train_y in train_ds:
    print(f'x = {train_x.numpy()}, y = {train_y.numpy()}')
print()

# 2. from_tensor_slices(x, y).shuffle(buffer size).batch(size)
"""
shuffle(buffer size) : tensor 행 단위로 셔플링
    - buffer size : 선택된 data size
batch size : 모델에 한 번에 공급할 dataset size
ex) 60000(MNIST) -> shuffle(10000).batch(100)
    1번째 slice data : 10000개 데이터 셔플링 -> 100개씩 추출
    2번째 slice data : 100개씩 추출
"""

# 1) x, y 변수 생성
x2 = tf.random.normal([5, 2])
y2 = tf.random.normal([5])

# 2) Dataset : 5개 slice -> 3 slice
train_ds2 = Dataset.from_tensor_slices((x, y)).shuffle(5).batch(2)
for train_x2, train_y2 in train_ds2:
    print(f'x = {train_x2.numpy()}, y = {train_y2.numpy()}')


# 3. keras 적용
from tensorflow.keras.datasets.cifar10 import load_data

# 1) dataset load
(x_train, y_train), (x_val, y_val) = load_data()

import matplotlib.pyplot as plt
# plt.imshow(x_train[0])  # 첫번째 이미지 확인
# plt.show()

# batch size = 100 image
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(100)

cnt = 0
for image_x, image_y in train_ds:
    cnt += 1
    print(f'image = {image_x.shape}, label = {image_y.shape}')
print(f'train slice 개수 = {cnt}')
# epochs = iter size(500) * batch size(100)

# val set batch size = 100 image
train_ds = Dataset.from_tensor_slices((x_val, y_val)).shuffle(1000).batch(100)
cnt = 0
for image_x, image_y in train_ds:
    cnt += 1
    print(f'image = {image_x.shape}, label = {image_y.shape}')
print(f'validation slice 개수 = {cnt}')
# epochs = iter size(100) * batch size(100)


"""
문) MNIST 데이터셋을 이용하여 train_ds, val_ds 생성하기
    train_ds : shuffle = 10000, batch size = 32
    val_ds : batch size = 32
"""
from tensorflow.keras.datasets.mnist import load_data
(x_train, y_train), (x_val, y_val) = load_data()

train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
val_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(32)
# 마지막에 남으면 앞에 있는거 끌어다 씀