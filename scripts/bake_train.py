# encode: utf-8
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
	y_train = []

	#0:muririn
	#1:kobuichi
	for i in range(0,2):
		path1 = "./train_data/kobuichi/"
		path2 = "./train_data/muririn/"
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
 		path1 = "./test_data/kobuichi/"
 		path2 = "./test_data/muririn/"
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

def train():
	x_train,y_train = getTrainDataSet()
	x_test,y_test = getTestDataSet()

	# x_train = chainer.Variable(x_train.reshape(1,2).astype(numpy.float32), volatile=False)
	# y_train = chainer.Variable(y_train.astype(numpy.int32), volatile=False)
	# x_test = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32), volatile=False)
	# y_test = chainer.Variable(y_data.astype(numpy.int32), volatile=False)
	x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3,250,250)) / 255
	y_train = np.array(y_train).astype(np.int32)
	x_test = np.array(x_test).astype(np.float32).reshape((len(x_test),3,250,250)) / 255
	y_test = np.array(y_test).astype(np.int32)

	model = clf_bake()
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	print(x_train.shape)
	epochNum = 5
	batchNum = 50
	epoch = 1

	while epoch <= epochNum:
		print("epoch:{}".format(epoch))
		print(datetime.datetime.now())

		trainImgNum = len(y_train)
		testImgNum = len(y_test)

		sumAcr = 0
		sumLoss = 0

		perm = np.random.permutation(trainImgNum)

		for i in six.moves.range(0,trainImgNum,batchNum):
			#ランダムにbatchNumの数だけ抽出
			x_batch = x_train[perm[i:i+batchNum]]
			y_batch = y_train[perm[i:i+batchNum]]

			optimizer.zero_grads()
			loss,acc = model.forward(x_batch,y_batch)
			loss.backward()
			optimizer.update()

			sumLoss += float(loss.data) * len(y_batch)
			sumAcr += float(acc.data) * len(y_batch)

		print('train mean loss={}, accuracy={}'.format(sumLoss / trainImgNum,sumAcr / trainImgNum))

		#テスト
		sumAcr = 0
		sumLoss = 0

		perm = np.random.permutation(testImgNum)

		for i in six.moves.range(0,testImgNum,batchNum):
			x_batch = x_test[perm[i:i+batchNum]]
			y_batch = y_test[perm[i:i+batchNum]]
			loss,acc = model.forward(x_batch,y_batch,train=False)

			sumLoss += float(loss.data) * len(y_batch)
			sumAcr += float(acc.data) * len(y_batch)

		print('test mean loss={}, accuracy={}'.format(sumLoss / testImgNum,sumAcr / testImgNum))

		epoch += 1

		S.save_hdf5('model' + str(epoch+1),model)

train()
