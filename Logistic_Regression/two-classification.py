# 多特征梯度下降算法
# 用于二分类的逻辑回归
import numpy as np
import matplotlib.pyplot as plt
# 学习率 非常关键的参数，调得不好容易导致无法收敛，甚至不断震荡至溢出
learning_rate = 0.001
# 迭代次数
iterations = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(x_data,y_data):
    weightslen = x_data.shape[1]
    weights = np.ones(weightslen)
    weights = np.mat(weights)
    for i in range(iterations):
        cost = sigmoid(x_data * weights.T)
        print(weights)
        weights = weights + learning_rate * (x_data.T * (y_data.T-cost)).T
        print(learning_rate * (x_data.T * (y_data.T-cost)).T)
    return weights


def printGraph(weights, x_data, y_data):
    class1x = []
    class1y = []
    class2x = []
    class2y = []
    minx = x_data[0][0]
    maxx = x_data[0][0]
    for i in range(0, x_data.shape[0]):
        if x_data[i][0]>maxx:
            maxx = x_data[i][0]
        if x_data[i][0]<minx:
            minx = x_data[i][0]
        if y_data[i] == 1:
            class1x.append(x_data[i][0])
            class1y.append(x_data[i][1])
        elif y_data[i] == 0:
            class2x.append(x_data[i][0])
            class2y.append(x_data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(class1x, class1y, s=30, c='red', marker='s')
    ax.scatter(class2x, class2y, s=30, c='green')
    pointx = np.arange(minx, maxx) # 用来画决策边界
    weights = np.array(weights)
    pointy = (-weights[0][2]-weights[0][0]*pointx)/weights[0][1]
    ax.plot(pointx, pointy)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def run():
    data_matrix = np.loadtxt('./data/data1.csv', skiprows=0)
    print("Input Matrix:")
    print(data_matrix)
    x_data = data_matrix[:, 0:-1]
    y_data = data_matrix[:, -1]
    # column 是0维的，增加一个常量
    scalar = np.ones(x_data.shape[0])
    x_data = np.column_stack((x_data, scalar))
    weights = gradient_descent(np.mat(x_data), np.mat(y_data))
    print("Weights:")
    print(weights)
    printGraph(weights, x_data, y_data)


if __name__ == '__main__':
    run()
