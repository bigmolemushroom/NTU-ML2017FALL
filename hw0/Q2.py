import sys
from PIL import Image

im = Image.open(sys.argv[1])
pix = im.load()
for i in range(0, im.size[0]):
	for j in range(0, im.size[1]):
		pix[i, j] = (pix[i, j][0] // 2, pix[i, j][1] // 2, pix[i, j][2] // 2)
im.save('Q2.png')
im.close()
