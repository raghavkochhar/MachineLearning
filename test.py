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


data = pd.read_csv("data/HR.csv")
datay = data['left']
datax = data.copy()
del datax['left']
print datax.columns

#One-Hot Encode categorical variables for scikit compantibility
datax = pd.get_dummies(datax, columns =['number_project','department','salary'], drop_first=True)

print datax.head()

#Set Train/Test Split Percentages
train_percent = np.arange(0.05, 1, 0.05)
test_percent = 1 - train_percent

#Create arrays to store train & validation accuracies
train_accuracy = np.zeros(len(train_percent))
test_accuracy = np.zeros(len(train_percent))

num_iterations = 1

for i in range(len(test_percent)):

    for j in range(num_iterations):
        x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size=test_percent[i], \
                                                            random_state=j)
        #Neural Nets
        #alpha is L2 penalty,
        #
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,15), activation= 'logistic', random_state=j)
        # clf.fit(x_train, y_train)
        #SVM
        clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
        train_accuracy[i] += clf.score(x_train, y_train)
        test_accuracy[i] += clf.score(x_test, y_test)

train_accuracy = train_accuracy/num_iterations
test_accuracy = test_accuracy/num_iterations

results_df = pd.DataFrame({'Train':train_accuracy, 'Test':test_accuracy}, index = train_percent)

print results_df
print("--- %s seconds ---" % (time.time() - start_time))


results_df.plot()
plt.show()






