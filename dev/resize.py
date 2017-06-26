import cv2
import numpy as np
import sys
import os

#(dirpath,savepath)

size = (250,250)
dirpath = sys.argv[1]
savepath = sys.argv[2]
name = sys.argv[3]

if dirpath[-1] != "/":
	dirpath = dirpath + "/"
if savepath[-1] != "/":
	savepath = savepath + "/"

if not os.path.exists(savepath):
		os.makedirs(savepath)

files = os.listdir(dirpath)

i = 0
for f in files:
	filename = dirpath + f
	img = cv2.imread(filename)
	i += 1
	resizeImg = cv2.resize(img,size)
	cv2.imwrite(savepath + "train_" + name + str(i) + ".jpg",resizeImg)
