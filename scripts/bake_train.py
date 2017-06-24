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
	x_test = []
	y_test = []

	#0:muririn
	#1:kobuichi
	for i in range(0,2):
		path1 = "../train_data/kobuichi/"
		path2 = "../train_data/muririn/"
		imgList1 = os.listdir(path1)
		imgList2 = os.listdir(path2)
		# print(imgList1)
		path = [path1,path2]
		imgLists = [imgList1,imgList2]
		imgList = imgLists[i]

		for j in range(len(imgList)):
			imgSrc = cv2.imread(path[i] + imgList[j])

			if imgSrc is None:
				continue
			x_train.append(imgSrc)
			y_train.append(i)

		return x_train,y_train

def getTestDataSet():
	x_test = []
	y_test = []
	for i in range(0,2):
 		path1 = "../test_data/kobuichi/"
 		path2 = "../test_data/muririn/"
 		imgList1 = os.listdir(path1)
 		imgList2 = os.listdir(path2)

 		path = [path1,path2]
 		imgLists = [imgList1,imgList2]
 		imgList = imgLists[i]

 		for j in range(len(imgList)):
 			imgSrc = cv2.imread(path[i] + imgList[j])
 			if imgSrc is None:
 				continue
 			x_test.append(imgSrc)
 			y_test.append(i)

 		return x_test,y_test
