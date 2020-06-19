"""
4_PythonII / chap07 / lecture04 / step03 내용과 동일 (경로만 변경)

1. csv file read
2. texts, target -> 전처리
3. max features
4. sparse matrix
5. train/test split
6. binary file save
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. csv file read
spam_data = pd.read_csv('C:/ITWILL/6_Tensorflow/data/temp_spam_data2.csv', encoding='UTF-8', header=None)

target = spam_data[0]
texts = spam_data[1]

# 2. texts, target 전처리

# 1) target 전처리
target = [1 if t == 'spam' else 0 for t in target]

# 2) texts : 영어는 벡터라이저에 불용어 처리 있음

# 3. max features
'''
사용할 x변수의 개수(열의 차수)
'''
tfidf_fit = TfidfVectorizer().fit(texts)
vocs = tfidf_fit.vocabulary_

# max_features = len(vocs)
max_features = 4000

# 4. sparse matrix
sparse_mat = TfidfVectorizer(stop_words='english', max_features=max_features).fit_transform(texts)  # 희소행렬
sparse_mat_arr = sparse_mat.toarray()

# 5. train/test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sparse_mat_arr, target, test_size=0.3)

# 6. numpy binary file save
import numpy as np
np.save('C:/ITWILL/6_Tensorflow/data/spam_data.npy', (x_train, x_test, y_train, y_test))  # 확장자 .npy

# np.save는 allow_pickle=True가 기본값인데 np.load는 False가 기본값
x_train, x_test, y_train, y_test = np.load('C:/ITWILL/6_Tensorflow/data/spam_data.npy', allow_pickle=True)

print(x_train.shape)
print(x_test.shape)