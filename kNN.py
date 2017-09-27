# coding:utf-8
'''
思路就是拿一组已经分类的数据集合，和要分类的数据进行计算距离
拿出来前k个距离这个数据最近的那些数据，出现次数最多的分类就是这个数据的分类。
'''

from numpy import *
import operator


def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, lables, k):
	# shape是计算array的大小多少航多少列
	dataSetSize = dataSet.shape[0]
	# 计算各个维度的值差距
	# array的+, -, * / 都是针对的每一个元素，而不是针对元素本身,
	# tile是的2个参数A, B表示A重复B次，B可以是一个元祖也可以是一个整数
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# 平方每个维度的差值
	sqDiffMat = diffMat ** 2
	# 计算平方和
	sqDistances = sqDiffMat.sum(axis=1)
	# 对平方和开平方
	distances = sqDistances**0.5
	# print distances
	# 排序,输出的是从小到大的元素所在的下标index
	sortedDistIndicies = distances.argsort()
	#print sortedDistIndicies
	classCount = {}
	for i in range(k):
		voteIlable = lables[sortedDistIndicies[i]]
		#print '--', voteIlable
		classCount[voteIlable] = classCount.get(voteIlable, 0) + 1

	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	#print sortedClassCount
	return sortedClassCount[0][0]

#g, l = createDataSet()
#a = classify0([0,0], g, l, 3)
#print a


def file2matrix(filename):
	fr = open(filename)
	arrayOnLines = fr.readlines()
	numberOfLines = len(arrayOnLines)
	returnMat = zeros((numberOfLines, 3))
	classLableVector = []
	index = 0
	for line in arrayOnLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLableVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLableVector


def autoNorm(dataSet):
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	#normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVal, (m, 1))
	# 这里的/不是矩阵除法，而是特征值相除
	normDataSet = normDataSet/tile(ranges, (m, 1))
	return normDataSet, ranges, minVal

def datingClassTest():
	hoRation = 0.1
	# 从文件中解析数据格式
	datingDataMat, datingLables = file2matrix('datingTestSet2.txt')	
	# 归一化处理
	normMat, ranges, minVals = autoNorm(datingDataMat)
	# 取出来10%的数字作为测试数据，90%作为样本数据
	m = normMat.shape[0]
	numTestVecs = int(m * hoRation)
	errorCount = 0
	for i in range(numTestVecs):
		# 循环10%的测试数据每一个进行分类
		classifierResult = classify0(normMat[i], normMat[numTestVecs:m,:], datingLables[100:1000], 3)
		print 'the classifier come back with: %d, the real answer is:%d' %(classifierResult, datingLables[i])
		if classifierResult != datingLables[i]:
			errorCount += 1
	print 'total error rate: %f' % (errorCount / float(numTestVecs))


def img2vector(filename):
	target = zeros((1, 1024))
	lr = open(filename)
	for i in range(32):
		lineStr = lr.readline()
		for j in range(32):
			target[0, i*32 + j] = int(lineStr[j])

	return target

from os import listdir
def handWriteingClassTest():
	# 训练源数据
	# 准备数据
	trainingFileList = listdir('trainingDigits')
	file_num = len(trainingFileList)
	trainingMat = zeros((file_num, 1024))	
	lables = []
	for index, filename in enumerate(trainingFileList):
		digit = int(filename.split('_')[0])
		trainingMat[index,:] = img2vector('trainingDigits/%s'% filename)
		lables.append(digit)
	
	# 组织训练数据
	test_file_list = listdir('testDigits')
	test_file_num = len(test_file_list)
	error_count = 0
	for index, filename in enumerate(test_file_list):
		real_digit = int(filename.split('_')[0])
		file_arrow = img2vector('testDigits/%s' % filename)
		classify_digit = classify0(file_arrow, trainingMat, lables, 3)
		print 'file:%s, real digit is :%s, clasify digit is :%s' %(filename, real_digit, classify_digit)		
		if int(real_digit) != classify_digit:
			error_count += 1
	print 'last error rate: %f' % (error_count / float(test_file_num))



	# 将训练出来的结果




