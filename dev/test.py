import numpy as np
import cv2

path = "../train_data/kobuichi/train1.jpg"
img = cv2.imread(path)
# test = range(100)
# test = np.array(range(100))
# test.reshape(len(test),2,2,2)
# test = np.arange(12).reshape(3,-1,-1)
print(img.astype(np.float32))
# x_train = chainer.Variable(x_train.reshape(1,2).astype(numpy.float32), volatile=False)
# print(len(img))
