from time import sleep
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import inspect
import random

n_points = 20000
confidence = 0.03
path_train='Train'

""" PRE-PROCESSING"""
# Read training data
onlyfiles_train = [ f for f in listdir(path_train) if isfile(join(path_train,f)) ]
images_train = np.empty(len(onlyfiles_train), dtype=object)

for n in range(0, len(onlyfiles_train)):
	images_train[n] = cv2.imread( join(path_train,onlyfiles_train[n]) )
	images_train[n] = cv2.resize(images_train[n],(150,150))


classes = []
training_data = []
# Model the data for ML
print("Creating training data...")
for img in images_train:
	print("#", end = '')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
	sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
	dst_max = dst.max()
	for i,row in enumerate(dst):
		for j,col in enumerate(row):
			image = np.zeros((3, 6, 1), np.float32)

			if col>confidence*dst_max and len(dst)-1 > i > 1 and len(row)-1 > j > 1:
				image[0][0] = sobelx[i-1][j-1]
				image[0][1] = sobelx[i-1][j]
				image[0][2] = sobelx[i-1][j+1]

				image[1][0] = sobelx[i][j-1]
				image[1][1] = sobelx[i][j]
				image[1][2] = sobelx[i][j+1]

				image[2][0] = sobelx[i+1][j-1]
				image[2][1] = sobelx[i+1][j]
				image[2][2] = sobelx[i+1][j+1]


				image[0][3] = sobely[i-1][j-1]
				image[0][4] = sobely[i-1][j]
				image[0][5] = sobely[i-1][j+1]

				image[1][3] = sobely[i][j-1]
				image[1][4] = sobely[i][j]
				image[1][5] = sobely[i][j+1]

				image[2][3] = sobely[i+1][j-1]
				image[2][4] = sobely[i+1][j]
				image[2][5] = sobely[i+1][j+1]



				
				classes.append(1)
				training_data.append(image)
				
			elif col<=confidence*dst_max and len(dst)-1 > i > 1 and len(row)-1 > j > 1:
				image[0][0] = sobelx[i-1][j-1]
				image[0][1] = sobelx[i-1][j]
				image[0][2] = sobelx[i-1][j+1]

				image[1][0] = sobelx[i][j-1]
				image[1][1] = sobelx[i][j]
				image[1][2] = sobelx[i][j+1]

				image[2][0] = sobelx[i+1][j-1]
				image[2][1] = sobelx[i+1][j]
				image[2][2] = sobelx[i+1][j+1]


				image[0][3] = sobely[i-1][j-1]
				image[0][4] = sobely[i-1][j]
				image[0][5] = sobely[i-1][j+1]

				image[1][3] = sobely[i][j-1]
				image[1][4] = sobely[i][j]
				image[1][5] = sobely[i][j+1]

				image[2][3] = sobely[i+1][j-1]
				image[2][4] = sobely[i+1][j]
				image[2][5] = sobely[i+1][j+1]


				
				classes.append(-1)
				training_data.append(image)
			
			
			


classes = np.array(classes, dtype=np.int32)
training_data = np.array(training_data, dtype=np.float32)
training_data = training_data.reshape( len(classes), 18)

print("\nDone!")
""" PRE-PORCESSING DONE """

print(len(training_data))
print(training_data)


# Train SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(0.1)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
svm.train(training_data[:n_points], cv2.ml.ROW_SAMPLE, classes[:n_points])

print(len(training_data[0]))
print("training done")
svm.save("SVM_MODEL_20K.xml")