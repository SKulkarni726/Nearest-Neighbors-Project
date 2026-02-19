import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"



column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]



iris = pd.read_csv(url, names=column_names)

sns.set(style="whitegrid")

pairplot = sns.pairplot(iris, hue="class", palette="husl", markers=["o", "s", "D"])

pairplot.fig.suptitle("Iris Species PairPlot", y=1.02)

plt.show()

X = iris.drop("class", axis=1)
y = iris["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy

cm = confusion_matrix(y_test, predictions, labels = knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
