'''
문3) iris.csv 데이터 파일을 이용하여 선형회귀모델  생성하시오.
     [조건1] x변수 : 2,3칼럼,  y변수 : 1칼럼
     [조건2] 7:3 비율(train/test set)
         train set : 모델 생성, test set : 모델 평가  
     [조건3] learning_rate=0.01
     [조건4] 학습 횟수 1,000회
     [조건5] model 평가 - MSE출력 
'''
import pandas as pd
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 
from sklearn.metrics import mean_squared_error # model 평가 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.model_selection import train_test_split # train/test set

iris = pd.read_csv('C:/ITWILL/6_Tensorflow/data/iris.csv')
print(iris.info())
cols = list(iris.columns)
iris_df = iris[cols[:3]]

# 1. x data, y data
x_data = iris_df[cols[1:3]] # x train
y_data = iris_df[cols[0]] # y tran

# 2. x,y 정규화(0~1) 
x_data = minmax_scale(x_data)

# 3. train/test data set 구성 
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# 4. 변수 정의
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

a = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.zeros(shape=[1]))

# 5. model 생성
model = tf.add(tf.matmul(X, a), b)
loss = tf.reduce_mean(tf.square(tf.subtract(Y, model)))

optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# 6. model training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(1000):
        _, loss_val = sess.run([train, loss], feed_dict={X: x_train, Y: y_train})
        a_val, b_val = sess.run([a, b])
        if not step % 100:
            print(f'step: {step+1}, MSE: {loss_val}')

    y_true = sess.run(Y, feed_dict={X: x_test, Y: y_test})
    y_pred = sess.run(model, feed_dict={X: x_test, Y: y_test})
    mse = mean_squared_error(y_true, y_pred)
    print(mse)