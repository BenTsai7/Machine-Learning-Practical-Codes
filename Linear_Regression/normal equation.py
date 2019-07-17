#标准方程法求解线性回归，时间复杂度O(n^3)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X_data矩阵m行 n+1列（n个特征，最后一列恒置为1)
# Y_data矩阵m行，m个目标值


def normal_equation(x_data, y_data):
    x_mat = np.mat(x_data)
    y_mat = np.mat(y_data).T
    xT = x_mat.T * x_mat
    if np.linalg.det(xT) == 0:
        print("cannot use normal equation because of matrix")
        return
    res = xT.I * x_mat.T * y_mat
    return res


def run():
    data_matrix = np.loadtxt('./data/test1.csv', delimiter=',', skiprows=1)
    print("Input Matrix:")
    print(data_matrix)
    x_data = data_matrix[:, 0:-1]
    y_data = data_matrix[:, -1]
    #column 是0维的，增加一个常量
    scalar = np.ones(x_data.shape[0])
    x_data = np.column_stack((x_data,scalar))
    weights = normal_equation(x_data, y_data)
    print("Weights:")
    print(weights)
    print("Error Caculation:")
    print(np.mat(x_data) * np.mat(weights)-np.mat(y_data).T)


if __name__ == '__main__':
    run()
