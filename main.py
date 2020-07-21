from time import sleep
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import inspect
from matplotlib import pyplot as plt



print("Loading SVM models...")

svm_5k = cv2.ml.SVM_load("Models/SVM_MODEL_20K_Shuffled_200px.xml")
svm_10k = cv2.ml.SVM_load("Models/SVM_MODEL_10K.xml")
svm_20k = cv2.ml.SVM_load("Models/SVM_MODEL_20K.xml")
svm_50k = cv2.ml.SVM_load("Models/SVM_MODEL_50K.xml")
svm_400k = cv2.ml.SVM_load("Models/SVM_MODEL_40K_Shuffled_400px.xml")

print("Models loaded!")


""" TESTING SVM CLASSIFIERS"""

path = "Testing/Canon_014_HR.png"

img_5k =cv2.imread(path)
img_10k =cv2.imread(path)
img_20k =cv2.imread(path)
img_50k =cv2.imread(path)
img_400k =cv2.imread(path)

img =  cv2.imread(path)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)

for i,row in enumerate(gray):
	for j, col in enumerate(row):
		image = np.zeros((3, 6, 1), np.float32)
		if len(gray)-1 > i > 1 and len(row)-1 > j > 1:
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


			image = image.flatten()

	

			if svm_5k.predict(np.ravel(image)[None, :])[1] == 1:
				img_5k = cv2.circle(img_5k, (j,i), 1, (0,0,255), 1) 

			if svm_10k.predict(np.ravel(image)[None, :])[1] == 1:
				img_10k = cv2.circle(img_10k, (j,i), 1, (0,0,255), 1) 

			if svm_20k.predict(np.ravel(image)[None, :])[1] == 1:
				img_20k = cv2.circle(img_20k, (j,i), 1, (0,0,255), 1) 

			if svm_50k.predict(np.ravel(image)[None, :])[1] == 1:
				img_50k = cv2.circle(img_50k, (j,i), 1, (0,0,255), 1) 

			if svm_400k.predict(np.ravel(image)[None, :])[1] == 1:
				img_400k = cv2.circle(img_400k, (j,i), 1, (0,0,255), 1) 




""" GET RESULTS FROM HARRIS FOR COMPARISON"""
img2 = cv2.imread(path)
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst2 = cv2.cornerHarris(gray,2,5,0.04)
img2[dst2>0.01*dst2.max()]=[0,255,0]


""" PLOT RESULTS """
f = plt.figure()
ax1 = f.add_subplot(2,3, 1)
ax1.title.set_text("HARRIS %1")
plt.imshow(img2[...,::-1])

ax2 = f.add_subplot(2,3, 2)
ax2.title.set_text("Classifier 1, 20K %5")
plt.imshow(img_5k[...,::-1], 'brg')


ax3 = f.add_subplot(2,3, 3)
ax3.title.set_text("Classifier 2, 10K %3")
plt.imshow(img_10k[...,::-1], 'brg')

ax4 = f.add_subplot(2,3, 4)
ax4.title.set_text("Classifier 3, 20K %3")
plt.imshow(img_20k[...,::-1],  'brg')

ax5 = f.add_subplot(2,3, 5)
ax5.title.set_text("Classifier 4 50K %3")
plt.imshow(img_50k[...,::-1], 'brg')

ax6 = f.add_subplot(2,3, 6)
ax6.title.set_text("Classifier 5 40K %5")
plt.imshow(img_400k[...,::-1], 'brg')


plt.show(block=True)

cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()