# coding:utf8
import operator
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
		# 并集
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
	

def bagOfWords2Vec(vocabList, inputSet):
	# 将输入的文档的单词集合(set)，拼装称一个0,1组合的向量
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
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
	p0 = sum(test_doc_vector * p_0_vector) + log(p_1_class)
	p1 = sum(test_doc_vector * p_1_vector) + log(1.0-p_1_class)
	if p1 > p0:
		return 1
	else:
		return 0


def testingNB():
	# 数据集合
	dataSet, labels = loadDataSet()
	# 单词向量
	vocabList = createVocabList(dataSet)
	# 矩阵
	train_matrix = []
	for data in dataSet:
		train_matrix.append(setOfWords2Vec(vocabList, data))
	p_0_vector, p_1_vector, p_1_class = trainNB0(train_matrix, labels)
	testEntry = ['love', 'my', 'dalmation']
	test_vector = array(setOfWords2Vec(vocabList, testEntry))
	print 'testEntry is :%s' % classifyNB(test_vector, p_0_vector, p_1_vector, p_1_class)
	test_entry = ['stupid', 'garbage']
	test_vector = array(setOfWords2Vec(vocabList, test_entry))
	print 'test_entry is %s' % classifyNB(test_vector, p_0_vector, p_1_vector, p_1_class)


def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [token.lower() for token in listOfTokens if len(token) > 2]


def spamTest():
	doc_list = []
	class_list = []
	for i in range(25):
		text = open('email/spam/%s.txt' % (i +1)).read()
		doc_list.append(textParse(text))
		class_list.append(1)
		text = open('email/ham/%s.txt' % (i + 1)).read()
		doc_list.append(textParse(text))
		class_list.append(0)
	doc_matrix = []
	vocab_list = createVocabList(doc_list)
	for doc_text in doc_list:
		doc_matrix.append(setOfWords2Vec(vocab_list, doc_text))
	
	test_list = []
	test_class = []
	train_list = []
	for i in range(10):
		index = int(random.uniform(0, len(doc_matrix)))
		test_list.append(doc_matrix[index])
		del doc_matrix[index]
		test_class.append(class_list[index])
		del class_list[index]
	print train_list, class_list
	p0, p1, p_class = trainNB0(array(doc_matrix), (class_list))
	error_count = 0.0
	for index, test_vector in enumerate(test_list):

		result = classifyNB(test_vector, p0, p1, p_class)
		if result != test_class[index]:
			error_count += 1
	print 'error rate:', error_count / 10

		
def calc_most_freq_words(vocab_list, full_text_words):
	freq_words = {}
	for word in vocab_list:
		freq_words[word] = full_text_words.count(word)
	result = sorted(freq_words.iteritems(), key=operator.itemgetter(1), reverse=True)
	return result[:30]


def text_parse_remove_stop(text):
	"""
	去掉停词
	"""
	stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

	import re
	listOfTokens = re.split(r'\W*', text)
	return [token.lower() for token in listOfTokens if len(token) > 2 and token not in stop_words]


def local_words(feed1, feed0):
	import feedparser
	doc_list = []
	class_list = []
	full_text = []
	min_count = min(len(feed1['entries']), len(feed0['entries']))
	# 构建单词矩阵，分类列表，并填充full_text表示所有的单词
	for i in range(min_count):

		word_list = text_parse_remove_stop(feed1['entries'][i]['summary'])
		doc_list.append(word_list)
		class_list.append(1)
		full_text.extend(word_list)
		
		word_list = text_parse_remove_stop(feed0['entries'][i]['summary'])
		doc_list.append(word_list)
		class_list.append(0)
		full_text.extend(word_list)
	# 构建词汇表
	
	vocab_list = createVocabList(doc_list)
	# print word_list

	# 去除出现次数最多的30个单词
	top_30_words = calc_most_freq_words(vocab_list, full_text)
	print top_30_words
	for word, count in top_30_words:
		vocab_list.remove(word)
	training_set = range(2 * min_count)
	test_set = []
	# 随机筛选出来20个文档作为测试文档，其他作为训练文档
	for i in range(20):
		index = int(random.uniform(0, len(training_set)))	
		test_set.append(index)
		del training_set[index]
	training_matrix = []
	training_classes = []
	for t_index in training_set:
		# 计算训练文档的向量
		vector =bagOfWords2Vec(vocab_list, doc_list[t_index])
		#training_classes.append(class_list[t_index])
		training_classes.append(class_list[t_index])
		training_matrix.append(vector)

	p0, p1, pc = trainNB0(array(training_matrix), training_classes)
	# 分类
	error_count = 0.0
	for doc_index in test_set:
		doc = doc_list[doc_index]
		vector = bagOfWords2Vec(vocab_list, doc)
		p = classifyNB(vector, p0, p1, pc)
		if p != class_list[doc_index]:
			error_count += 1
	return p0, p1, error_count / 20, vocab_list

		
