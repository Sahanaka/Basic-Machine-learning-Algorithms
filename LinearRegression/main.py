# Linear Regression model

import pandas as pd
import numpy as np
import sklearn # scikit learn package
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read the data sets
data = pd.read_csv('./Datasets/student/student-mat.csv', sep=";")
print(data.head())

# Trim the data set
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # Attributes
print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))  # Attributes
y = np.array(data[predict]) # labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,  test_size = 0.1)

best_accuracy = 0

"""
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,  test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    # Saves the model if the current accuracy is the best
    if (accuracy > best_accuracy):
        best_accuracy = accuracy
        with open('studentmodel.pickle', 'wb') as file:
            pickle.dump(linear, file) """

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in) # Load the saved model

# Printing the coefficients of the best fit line
print("Co: ", linear.coef_) # 5 values gives the slopes in 5 dimensions
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) # Compare the predictions with actual value

# Plotting
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

