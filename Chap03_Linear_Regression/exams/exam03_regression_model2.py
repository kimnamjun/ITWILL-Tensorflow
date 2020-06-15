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

iris = pd.read_csv('D:/ITWILL/6_Tensorflow/data/women.csv')
print(iris.info())
cols = list(iris.columns)
iris_df = iris[cols[:3]] 

# 1. x data, y data
x_data = iris_df[cols[1:3]] # x train
y_data = iris_df[cols[0]] # y tran

# 2. x,y 정규화(0~1) 
x_data = minmax_scale(x_data)


# 3. train/test data set 구성 


# 4. 변수 정의


# 5. model 생성 


# 6. model training
