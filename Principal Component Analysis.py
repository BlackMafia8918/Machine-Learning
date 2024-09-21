## Importing support libraries
from __future__ import print_function
from time import time
import numpy as np ## for numerical calculations of arrays
import pandas as pd ## for reading csv file and wroking with dataframe operations
from PIL import Image ## for image processing and output
import cv2 ## for image processing
from numpy import linalg as LA ## numpy's Linear Algebra library
import os ## for reading images from image folder
import math
from scipy.spatial import distance ## for calculating distance between two entities
import requests ## to get the file from url
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

images = np.load('olivetti_faces.npy')
target = np.load('olivetti_faces_target.npy')
print(images.shape, target.shape)

from skimage.io import imshow
loadImage = images[13]
imshow(loadImage) 

loadImage

total_images = 400
total_classes = 10

data = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target)

n_comp = 10
pca = PCA(n_components=n_comp, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

number_of_eigenfaces=len(pca.components_)
eigen_faces=pca.components_.reshape((number_of_eigenfaces, images.shape[1], images.shape[2]))

a=np.zeros((64,64))

cols=10
rows=int(number_of_eigenfaces/cols)
fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
axarr=axarr.flatten()
for i in range(number_of_eigenfaces):
    axarr[i].imshow(eigen_faces[i],cmap="bone")
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title("eigen id:{}".format(i))
    a=a+eigen_faces[i]
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))

np.shape(eigen_faces)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_pca)
print(accuracy_score(y_test, y_pred))

fig,ax=plt.subplots(1,1,figsize=(8,8))
ax.imshow(pca.mean_.reshape((64,64)), cmap="bone")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Average Face')

fig, axarr=plt.subplots(nrows=40, ncols=6, figsize=(100, 200))
axarr=axarr.flatten()
i = 0
for x,y in zip(y_test,y_pred):
    x=x*10
    y=y*10
    axarr[i].imshow(images[x],cmap='gray')
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title("Actual - face id:  {}".format(x), fontsize=50)
    i=i+1
    axarr[i].imshow(images[y],cmap='gray')
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title(" Predicted - face id: {}".format(y), fontsize=50)
    i = i+1

plt.show()
