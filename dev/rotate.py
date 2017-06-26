import os
import sys
from PIL import Image

dirpath = sys.argv[1]

if dirpath[-1] != "/":
	dirpath = dirpath + "/"

files = os.listdir(dirpath)


for f in files:
	img = Image.open(dirpath + f)

	img = img.transpose(Image.FLIP_TOP_BOTTOM)
	img.save(dirpath + "180_" + f)

	img = img.transpose(Image.ROTATE_90)
	img.save(dirpath + "90_" + f)

	img = img.transpose(Image.ROTATE_270)
	img.save(dirpath + "270_" + f)
