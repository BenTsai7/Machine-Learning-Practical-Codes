# 单特征二维x,y平面的对数线性梯度下降算法
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
# 学习率 非常关键的参数，调得不好容易导致无法收敛，甚至不断震荡至溢出
learning_rate = 0.01
# 迭代次数
iterations = 10000


def gradient_descent(points):
    a = 0
    b = 0
    pointnum = float(len(points))
    for i in range(iterations):
        x_gradient = 0
        y_gradient = 0
        for j in range(0, len(points)):
            x = points[j, 0]
            y = points[j, 1]
            x_gradient += ((a*x+b-y)*x)
            y_gradient += (a*x+b-y)
        x_gradient = x_gradient / pointnum
        y_gradient = y_gradient / pointnum
        a = a - learning_rate * x_gradient
        b = b - learning_rate * y_gradient
    return [a, b]


def run():
    points = genfromtxt('./data/Salary_Data.csv', delimiter=',', skip_header=1)
    data = pd.read_csv('./data/Salary_Data.csv', names=['x', 'y'], header=0)
    for j in range(0, len(points)):
        points[j, 1] = log(points[j, 1])
    [a, b] = gradient_descent(points)
    plt.plot(data.x, data.y, 'bo')
    plt.plot(data.x, exp(data.x * a+b))
    plt.show()


if __name__ == '__main__':
    run()
