import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')


# data = pd.read_csv("data/HR.csv")
# y = data['left']


iris = datasets.load_iris()

#Set Train/Test Split Percentages
train_percent = np.arange(0.05, 1, 0.05)
test_percent = 1 - train_percent

#Create arrays to store train & validation accuracies
train_accuracy = np.zeros(len(train_percent))
test_accuracy = np.zeros(len(train_percent))

num_iterations = 10

for i in range(len(test_percent)):

    for j in range(num_iterations):
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_percent[i], \
                                                            random_state=j)
        #Neural Nets
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(x_train, y_train)
        #SVM
        # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
        train_accuracy[i] += clf.score(x_train, y_train)
        test_accuracy[i] += clf.score(x_test, y_test)

train_accuracy = train_accuracy/num_iterations
test_accuracy = test_accuracy/num_iterations


results_df = pd.DataFrame({'Train':train_accuracy, 'Test':test_accuracy},index = train_percent)

print results_df

results_df.plot()
plt.show()






