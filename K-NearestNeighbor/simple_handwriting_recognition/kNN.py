import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

#将32*32的二进制图像转换为1*1024向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


#手写数字分类测试
def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFilesList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFilesList)
    # 初始化训练的Mat矩阵（全零阵），测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFilesList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构造kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型，trainingMat为测试矩阵，hwLabels为对应标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("predict: %d real: %d " % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("error data number:%d\nerror rate:%f%%" % (errorCount, errorCount/mTest * 100))
        

def main():
    handwritingClassTest()
    
    
if __name__ == '__main__':
    main()
