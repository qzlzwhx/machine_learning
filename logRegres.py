# coding:utf8

from numpy import *


def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('./Ch05/testSet.txt')
	for data in fr.readlines():
		line = data.strip().split()
		dataMat.append([1.0, float(line[0]), float(line[1])])
		labelMat.append(int(line[2]))
	return dataMat, labelMat


def sigmoid(intX):
	return 1.0 / (1 + exp(-intX))


def gradAscent(data_array, class_labels):
	# 转化成矩阵
	data_matrix = mat(data_array)
	# 转化称列向量
	class_labels = mat(class_labels).transpose()
	m, n = shape(data_matrix)
	# 移动距离
	alpha = 0.001
	max_cycles = 500
	weights = ones((n, 1))
	# 矩阵乘，由左边的矩阵绝对行，右边的矩阵绝对列
	# 所谓，乘法结果，第m行n列那个元素就是左边矩阵第m行，和右边矩阵第n列挨个相乘之和所得
	# 所谓
	for k in range(max_cycles):
		h = sigmoid(data_matrix * weights)
		error = class_labels - h
		weights = weights + alpha * data_matrix.transpose() * error
	return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def stocGradAscent0(data_mat, class_labels):
	#data_matrix = mat(data_mat)
	row, column = shape(data_mat)
	alpha = 0.01
	weights = ones(column)
	for i in range(row):
		h = sigmoid(sum(data_mat[i] * weights))
		error = class_labels[i] - h
		weights = weights + alpha * data_mat[i] * error
	return weights



def stocGradAscent1(dataMat, labels, times=150):
	row, column = shape(dataMat)
	weights = ones(column)
	for j in range(times):
		dataIndex = range(row)
		for i in range(row):

			alpha = 4.0 / (i + j + 1.0) + 0.01
			rand_index = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMat[rand_index] * weights))
			error = labels[rand_index] - h
			weights = weights + alpha * error * dataMat[rand_index]
			del dataIndex[rand_index]
	return weights



