from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


def run():
    points = genfromtxt('./data/Salary_Data.csv', delimiter=',', skip_header=1)
    data = pd.read_csv('./data/Salary_Data.csv', names=['x', 'y'], header=0)
    dtr = tree.DecisionTreeRegressor()
    dtr = dtr.fit(mat(points[:, 0].reshape(-1, 1)), data.y)
    plt.plot(data.x, data.y, 'bo')
    plt.plot(data.x, dtr.predict(reshape(data.x.tolist(), (-1, 1))))
    plt.show()


if __name__ == '__main__':
    run()
