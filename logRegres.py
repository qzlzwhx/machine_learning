# coding:utf8
"""
逻辑回归的主要思想就是，根据现有数据对分类边界线建立回归公式，以此为分类。训练分类器的过程就是寻找最佳拟合参数的过程。
sigmoid函数，单位阶跃函数的一种。这种函数在x=0的时候，sigmoid结果是0.5，随着x无限增大，sigmoid越来越接近与1，想法无限减小的时候在逐渐
接近0。而逻辑回归算法就是为了寻找这一样一个最佳系数，sigmoid的输入z由如下公式得到，z=wx,w是一个列向量，就是我们寻找的回归系数，
实际上他就是权重，每个特征的权重，x就是输入数据，一个一个的元素。
所以寻找回归系数的过程就是，计算每一个输入单元，对权重的影响，最终得到你想要的权重值，也就是回归系数。
"""

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
	# 注意和下边随机梯度上升算法的区别，
	for k in range(max_cycles):
		# 计算整个矩阵所有元素的sigmoid值，也就是100*3的矩阵和3*1的矩阵相乘，得到100*1的列向量
		# 这个h就是对矩阵元素的预测，因为预测结果就是根据sigmoid来的吗
		h = sigmoid(data_matrix * weights)
		# 两个列向量减法，这里的error的意思就是真是标签和预测标签的差值
		error = class_labels - h
		# 计算整个矩阵对权重的影响，也就是回归系数。因为error是一个100*1的向量，所以需要对矩阵进行转置3*100
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
		# 随机梯度上升，这里计算是挨个元素计算，h的意义就是下表是i的那个元素，的sigmoid值，也就是它的预测分类
		h = sigmoid(sum(data_mat[i] * weights))
		# 真是分类和预测分类的差值
		error = class_labels[i] - h
		# 权重，计算，该元素对权重带来的影响
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


def classifyVector(intX, weights):
	prob = sigmoid(sum(intX*weights))
	if prob > 0.5:
		return 1
	else:
		return 0


def colicTest():
	training_file = open('./Ch05/horseColicTraining.txt')
	test_file = open('./Ch05/horseColicTest.txt')
	data_list = []
	labels = []
	for line in training_file.readlines():
		attrs = line.strip().split('\t')
		data_list.append(map(float, attrs[:20]))
		labels.append(float(attrs[-1]))
	# print data_list
	weights = stocGradAscent1(array(data_list), labels, 500)
	error_count = 0
	num_test = 0
	for line in test_file.readlines():
		num_test += 1
		atrrs = line.strip().split('\t') 
		real_label = int(atrrs[21])
		if classifyVector(map(float, attrs[:20]), weights) != real_label:
			error_count += 1
	return error_count, num_test, weights
			

def multitest(times=10):
	
	for i in range(10):
		a, b,c = colicTest()
		print a, b, c
			

		
		
