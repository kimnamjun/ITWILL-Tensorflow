"""
csv(pandas object) -> tensorflow variable
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

iris = pd.read_csv('C:/ITWILL/6_Tensorflow/data/iris.csv')
iris.info()

# 공급 data 생성
cols = list(iris.columns)
x_data = iris[cols[:4]]
y_data = iris[cols[-1]]

# X, Y 변수 정의 : tensorflow 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.string, shape=[None])

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# session object : data 공급 -> 변수
with tf.Session() as sess:
    # 훈련용 data 공급
    feed_data = {X: x_train, Y: y_train}
    X_val, Y_val = sess.run([X, Y], feed_dict=feed_data)

    # 평가용 data 공급
    feed_data = {X: x_test, Y: y_test}
    X_val2, Y_val2 = sess.run([X, Y], feed_dict=feed_data)

    # numpy -> pandas 변경
    X_df = pd.DataFrame(X_val2, columns=['a','b','c','d'])
    Y_ser = pd.Series(Y_val2)
    print(Y_ser.value_counts())
