import sklearn.neural_network as sk_nn
from sklearn.model_selection import train_test_split
import numpy as np

dataset = np.loadtxt('data/test1.csv', delimiter=",", skiprows=1)
X = dataset[:, 0:-1]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = sk_nn.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                            learning_rate_init=0.001, max_iter=200)
model.fit(X_train, y_train)
accuracy = model.score(X_test,y_test)
print("accuracy: ", accuracy)