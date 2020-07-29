import sklearn
from sklearn import datasets
from sklearn import svm # Classifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Load data sets from sklearn
cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benign"]

clf = svm.SVC(kernel="linear", C=1) # Support vector classification
clf.fit(x_train, y_train)

clf2 = KNeighborsClassifier(n_neighbors=9)
clf2.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred2 = clf2.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy2 = metrics.accuracy_score(y_test, y_pred2)
print(accuracy, accuracy2) # Comparing with KNN