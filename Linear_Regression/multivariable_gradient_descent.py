# 多特征线性梯度下降算法
import numpy as np
import matplotlib.pyplot as plt
# 学习率 非常关键的参数，调得不好容易导致无法收敛，甚至不断震荡至溢出
learning_rate = 0.01
# 迭代次数
iterations = 5000
#误差曲线变量
global error_x
global error_y
global count
error_x = []
error_y = []
count = 0


def update_error(weights, x_data, y_data):
    error_matrix = np.mat(x_data) * np.mat(weights).T - np.mat(y_data).T
    error = 0.0
    for i in range(error_matrix.shape[0]):
        error += error_matrix.tolist()[i][0]**2
    error /= error_matrix.shape[0]
    global count
    global error_x
    global error_y
    error_x.append(count)
    error_y.append(error)
    count += 1


def gradient_descent(x_data,y_data):
    weightslen = x_data.shape[1]
    weights = np.zeros(weightslen)
    pointnum = float(x_data.shape[0])
    for i in range(iterations):
        gradients = np.zeros(weightslen)
        for j in range(0, x_data.shape[0]):
            for k in range(0, weightslen):
                gradients[k] += (np.mat(x_data[j])*np.mat(weights).T-y_data[j])*x_data[j][k]
        for k in range(0,weightslen):
            gradients[k] /= pointnum
            weights[k] -= learning_rate * gradients[k]
        update_error(weights, x_data, y_data)
    return weights


def run():
    data_matrix = np.loadtxt('./data/test1.csv', delimiter=',', skiprows=1)
    print("Input Matrix:")
    print(data_matrix)
    x_data = data_matrix[:, 0:-1]
    y_data = data_matrix[:, -1]
    # column 是0维的，增加一个常量
    scalar = np.ones(x_data.shape[0])
    x_data = np.column_stack((x_data, scalar))
    weights = gradient_descent(x_data, y_data)
    print("Weights:")
    print(weights)
    print("Error Caculation:")
    print(np.mat(x_data) * np.mat(weights).T - np.mat(y_data).T)
    print(error_y)
    plt.plot(np.array(error_x), np.array(error_y))
    plt.show()


if __name__ == '__main__':
    run()
