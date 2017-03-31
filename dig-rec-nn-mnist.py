## Digit Recognition using Neural Network 


## Import modules
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neural_network, metrics
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

## Load data
ini_t = time()
features_filedata = np.loadtxt('mnist_test.csv', delimiter = ',')
label_filedata = np.loadtxt('mnist-label-test.csv', delimiter = ',')

## Labels
y = label_filedata

## Features
X = features_filedata

## Convert to hog
hog_list = []
for i in X:
    hf = hog(i.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(7, 7), visualise=False)
    hog_list.append(hf)
hog_data = np.array(hog_list, 'float')

## Processing the data
scaler = StandardScaler().fit(hog_data)
hog_data = scaler.transform(hog_data)

## Split data
X_train, X_test, y_train, y_test = train_test_split(hog_data, y, test_size=0.2, random_state = 25)

## Train the neural network
classifier = neural_network.MLPClassifier(solver='sgd',tol=1e-4, max_iter = 400, alpha = 1e-4, hidden_layer_sizes = (30,),random_state=1)
classifier.fit(X_train, y_train.ravel())

## Run the classifier on test data
pred = classifier.predict(X_test)

## Classifier report
print ("Report")
print metrics.classification_report(y_test, pred)

print('Running time: %0.3fs' % (time() - ini_t))