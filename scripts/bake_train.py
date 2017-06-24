import cv2
import os
import six
import datetime

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from clf_bake_model import clf_bake

import numpy as np


def getTrainDataSet():
	x_train = []
	x_test = []
	y_train = []
	y_test = []

	#0:muririn
	#1:kobuichi
	for i in range(0,2):
		path1 = "/home/ai/Documents/python/yuzuIce/train_data/kobuichi/"
		path2 = "/home/ai/Documents/python/yuzuIce/train_data/muririn/"
		test_path = "/home/ai/Documents/python/kobu_muri/learn_data/a"
		imgList1 = os.listdir(path1)
		imgList2 = os.listdir(path2)
		# print(imgList1)
		path = [path1,path2]
		imgLists = [imgList1,imgList2]
		imgList = imgLists[i]

		for j in range(len(imgList)):
			imgSrc = cv2.imread(path[i] + imgList[j])
			print(imgSrc)
			# print(f)

getTrainDataSet()
