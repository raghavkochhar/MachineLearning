import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import copy
import matplotlib
import time

matplotlib.style.use('ggplot')

#Start Timer
start_time = time.time()


data = np.array([[0,0,0],[1,1,1],[2,2,0],[3,3,1]])
data = pd.DataFrame(data, columns= ["A","B",'left'])


data = data.iloc[np.random.permutation(len(data))]

print data

data = data.sort_values(by=['left'])

data50 =  pd.concat([data.iloc[:1,:], data.iloc[-1:,:]])

print data50

print data['left'].sum(axis=0)