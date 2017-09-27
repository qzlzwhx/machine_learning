# coding:utf8
'''
思路就是从所有特征中找到最优的分类特征，进行递归创造树
如果特征A的某个值分类之后所有的分类都是一样的，那么这个特征A的该值的分类应该就是这个分类。
如果该值分类之后已经没有其余特征可以使用了，那么选取出现此处最多的分类作为该值的分类。
如果某个值分类之后还有分类，那么表示还需要进行递归处理上述逻辑

'''
import pickle
from math import log


def calcShannonEat(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for data in dataSet:
		current_label = data[-1]
		if current_label not in labelCounts.keys():
			labelCounts[current_label] = 0
		labelCounts[current_label] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = labelCounts.get(key) / float(numEntries)
		#print prob * log(prob, 2), log(prob, 2)
		shannonEnt -= prob * log(prob,2)
	return shannonEnt


def createDataSet():
	dataSet = [
		[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


def splitDataSet(dataSet, axis, value):
	resetDataSet = []
	for data in dataSet:
		temp_data = data[:]
		if temp_data.pop(axis) == value:
		
			resetDataSet.append(temp_data)
	return resetDataSet


def chooseBestFeatureToSplit(dataSet):
	num_features = len(dataSet[0]) - 1
	
	best_shannon_ent = calcShannonEat(dataSet)
	index = 0
	num_entries = len(dataSet)
	for i in range(num_features):
		# 取出该列所有特征值
		feature_values = set([data[i] for data in dataSet])
		# feature_values = set(feature_values)
		# 根绝列值也就是特征值划分称不同子集，计算综合熵
		sub_shannon = 0.0
		for feature_value in feature_values:
			# 根据特征值划分数据集合
			subSet = splitDataSet(dataSet, i, feature_value)
			# 子集合的概率
			subSet_prob = len(subSet) / float(num_entries)
			# 根据i列进行划分后香浓熵的，每个子集合的熵的和
			sub_shannon += subSet_prob * calcShannonEat(subSet)
		if sub_shannon < best_shannon_ent:
			index = i
			best_shannon_ent = sub_shannon
	return index


def majorityCnt(class_list):
	single = set(class_list)
	class_count = dict((c, 0) for c in single)
	for value in class_list:
		class_count[value] += 1
	sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
	
	return sorted_class_count[0][0]


def create_tree(dataSet, labels):
	# 如果数据集合中所有的标签都相同，那么表示这个特征值就应该是这个标签了
	class_list = [data[-1] for data in dataSet]
	if class_list.count(class_list[0]) == len(class_list):
		return class_list[0]
	# 如果数据集合只有一列了，那么表示就只剩下最后一列分类列了，所以就不再迭代了，返回class_list里边出现次数最多的分类
	if len(dataSet[0]) == 1:
		return majorityCnt(class_list)

	# 获取数据集合中的最优的特征
	bestFeatIndex = chooseBestFeatureToSplit(dataSet)
	feature_name = labels.pop(int(bestFeatIndex))
	# 获取特征都有那些值，以继续分割
	feat_values = set([ex[bestFeatIndex] for ex in dataSet])
	# 构建一个树，树的第一个根元素也就是第一个分割数据集合的特征名字feature_name，他也是一个字典，
	# 他所拥有的是，它的各个值所对应的分类到底是什么，以此递归
	my_tree = {feature_name:{}}
	for feat_value in feat_values:
		sublabels = labels[:]
		sub_set = splitDataSet(dataSet, bestFeatIndex, feat_value)
		my_tree[feature_name][feat_value] = create_tree(sub_set, sublabels)

	return my_tree

	
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	feature_index = featLabels.index(firstStr)
	test_feature_value = testVec[feature_index]
	for feature_value in secondDict:
		if feature_value == test_feature_value:
			if isinstance(secondDict[feature_value], dict):
				classLabel = classify(secondDict[feature_value], featLabels, testVec)
			else:
				classLabel = secondDict[feature_value]
	return classLabel


def storeTree(input_tree, filename):
	fw = open(filename, 'w')
	pickle.dump(input_tree, fw)
	fw.close()


def grab_tree(filename):
	fr = open(filename)
	data = pickle.load(fr)	
	fr.close()
	return data

