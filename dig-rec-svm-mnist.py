## Digit Recognition using SVM and MNIST data

## Import modules
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from skimage.feature import hog
from sklearn.preprocessing import normalize

## Load data
ini_t = time()
filedata = np.loadtxt('mnist_test.csv', delimiter = ',')
filedata1 = np.loadtxt('mnist-label-test.csv', delimiter = ',')

## Labels
y = filedata1

## Features
X = filedata


# Convert to hog
hog_list = []
for i in X:
	hf = hog(i.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(7, 7), visualise=False)
	print hf.shape
	hog_list.append(hf)
hog_data = np.array(hog_list, 'float')

## Normalize the data
hog_data = normalize(hog_data) 

## Split data
X_train, X_test, y_train, y_test = train_test_split(hog_data, y, test_size=0.2, random_state = 25)

## Train SVM Classifier
classifier = svm.LinearSVC()
classifier.fit(X_train, y_train.ravel())

## Run the classifier on test data
pred = classifier.predict(X_test)

## Classifier report
print ("Report")
print metrics.classification_report(y_test, pred)

print('Running time: %0.3fs' % (time() - ini_t))