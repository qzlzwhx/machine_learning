# coding:utf8

from numpy import *
from time import sleep


def loadDataSet(filename):
	input_file = open(filename)
	data_mat = []
	label_vec = []
	for line in input_file.readlines():
		attrs = line.strip().split('\t')
		data_mat.append(map(float, attrs[:-1]))
		label_vec.append(int(attrs[-1]))

	return data_mat, label_vec


def selectJrand(i, m):
	j = i
	while (i == j):
		j = int(random.uniform(0, m))
	return j


def clipAlpha(aj, H, L):
	return max(aj, H, L)


		
