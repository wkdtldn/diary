# 출처 : https://buomsoo-kim.github.io/keras/2019/07/01/Easy-deep-learning-with-Keras-18.md/

import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# parameters for data load
num_words = 30000
maxlen = 50
test_split = 0.3

# load data
print('\n' + '-----line 17-----')
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)
print(X_train + '\n' + y_train + '\n' + X_test + '\n' + y_test + '\n')


# padding
print('\n' + '-----line 23-----')
X_train = pad_sequences(X_train, padding = 'post')
print(X_train)
print('\n' + '-----line 26-----')
X_test = pad_sequences(X_test, padding = 'post')
print(X_test)
print('\n')


# reshape
print('\n' + '-----line 33-----')
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
print(X_train)
print('\n' + '-----line 36-----')
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
print(X_test)
print('\n')


# y_data Organize
print('\n' + '-----line 43-----')
y_data = np.concatenate((y_train, y_test))
print(y_data)
print('\n' + '-----line 46-----')
y_data = to_categorical(y_data)
print(y_data)
print('\n')


# 더 검색이 필요
print('\n' + '-----line 53-----')
y_train = y_data[:1395]
print(y_train)
print('\n' + '-----line 56-----')
y_test = y_data[1395:]
print(y_test)
print('\n')


# 데이터의 모양 출력하기
print('\n' + '-----line 63.66-----')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)