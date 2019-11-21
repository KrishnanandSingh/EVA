import os
from PIL import Image

path = '/home/krishnanand/Downloads/images/'
outpath = '/home/krishnanand/Downloads/images_modified/'
counter = 1
for r, d, f in os.walk(path):
	for file in f:
		print(file)
		im = Image.open(path + file)
		im = im.resize((400,400))
		if counter < 10:
			im.save(outpath + 'img_00' + str(counter)+'.jpg')
		elif counter <100:
			im.save(outpath + 'img_0' + str(counter)+'.jpg')
		else:
			im.save(outpath + 'img_' + str(counter)+'.jpg')
		print (counter)
		counter = counter + 1
