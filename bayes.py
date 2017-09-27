# coding:utf8
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataSet):
	"""
	创造一个所有单词组成的向量，元素就是单词
	"""
	vocabSet = set([])
	for data in dataSet:
		vocabSet = vocabSet | set(data)
	return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
	# 将输入的文档的单词集合(set)，拼装称一个0,1组合的向量
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print 'word {} not in vocabList'.format(word)
	return returnVec
	


def trainNB0(trainMatrix, labels_list):
	
	num_docs = len(trainMatrix)
	words_num = len(trainMatrix[0])
	# 初始化为1是为了避免概率是0的情况，因为后边计算的时候是乘积。并且分母初始化成2
	p_0_vector = ones(words_num)
	p_1_vector = ones(words_num)
	p_0_words_count = 2.0
	p_1_words_count = 2.0
	for index, vector in enumerate(trainMatrix):
		if labels_list[index] == 1:
			p_1_vector += vector
			p_1_words_count += sum(vector)
		else:
			p_0_vector += vector
			p_0_words_count += sum(vector)
	# 由于python有精度问题，所以会出现下溢问题，即2个非常小的小数乘积=0，所以这里采用对数
	prob_0_vector = log(p_0_vector / p_0_words_count)
	prob_1_vector = log(p_1_vector / p_1_words_count)
	return prob_0_vector, prob_1_vector, sum(labels_list) / float(len(labels_list))


def classifyNB(test_doc_vector, p_0_vector, p_1_vector, p_1_class):
	p0 = test_doc_vector * p_0_vector + log(p_1_class)
	p1 = test_doc_vector * p_1_vector + log(1.0-p_1_class)	
	if p1 > p0:
		return 1
	else:
		return 0




