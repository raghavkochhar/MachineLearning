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



def HR_data():
    # # INCLUDE DEPARTMENTS Analytics
    data = pd.read_csv("data/HR.csv")

    # Due to the data having 15k instances and being unmanageable. The data will be split into 50/50
    # positive/negative samples to reduce the instances to 7k

    data = data.iloc[np.random.permutation(len(data))]
    data = data.sort_values(by=['left'])
    num_pos = data['left'].sum(axis=0)

    data50 = pd.concat([data.iloc[:num_pos,:], data.iloc[-num_pos:,:]])

    #Keep
    data50 = pd.read_csv("data/HR.csv")

    #Create X and Y datsets
    datay = data50['left']
    datax = data50.copy()
    del datax['left']
    del datax['department']

    print datax.columns

    #One-Hot Encode categorical variables for scikit compantibility
    datax = pd.get_dummies(datax, columns =['number_project','salary'], drop_first=True)


    # # INCLUDE DEPARTMENTS
    # del datax['left']
    # datax = pd.get_dummies(datax, columns =['number_project','department','salary'], drop_first=True)

    return datax, datay




def wine_data():

    data = pd.read_csv("MachineLearning/data/winequality-red.csv")

    datay =  data['quality']
    datax = data.copy()
    del datax['quality']

    return datax, datay


def learning_curve(datax, datay, num_iterations):

    # Set Train/Test Split Percentages
    train_percent = np.arange(0.05, 0.95, 0.05)
    test_percent = 1 - train_percent

    # Create arrays to store train & validation accuracies
    train_accuracy = np.zeros(len(train_percent))
    test_accuracy = np.zeros(len(train_percent))


    for i in range(len(test_percent)):

        for j in range(num_iterations):
            x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size=test_percent[i], \
                                                                random_state=j)

            print "------------%s---------------" % (train_percent[i])
            print "TRAIN COUNT: %s/ %s" %(sum(y_train), len(y_train))
            print "TEST COUNT: %s/%s" % (sum(y_test), len(y_test))

            #Neural Nets
            #alpha is L2 penalty,
            #
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation= 'logistic', \
                                hidden_layer_sizes=(100,), random_state=j, \
                                early_stopping = True).fit(x_train, y_train)

            #SVM

            # clf = svm.SVC(C=1, kernel ='rbf', random_state=j).fit(x_train, y_train)
            train_accuracy[i] += clf.score(x_train, y_train)
            test_accuracy[i] += clf.score(x_test, y_test)

    train_accuracy = train_accuracy/num_iterations
    test_accuracy = test_accuracy/num_iterations


    return pd.DataFrame({'Train':train_accuracy, 'Test':test_accuracy}, index = train_percent)


# Run HR Analytics

datax, datay = wine_data()

# datax, datay = HR_data()
results_df = learning_curve(datax, datay, num_iterations = 10)


print("--- %s seconds ---" % (time.time() - start_time))


print results_df
results_df.plot()
plt.show()

#





