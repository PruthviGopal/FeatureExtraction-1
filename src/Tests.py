'''
Created on 08.01.2014

@author: b4mbi
'''
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import pylab as pl
from AKFA import akfa
import numpy as np
from scipy import sparse
#import gc
import Util
import akfa as feature
from scipy.spatial.distance import cdist
import h5py as hpy
import time
from scipy.spatial.distance import pdist, squareform
from scipy import exp
import scipy as sp
from scipy import spatial
from scipy.sparse import bmat
from scipy.sparse import csr_matrix
import math
from sklearn.metrics.pairwise import pairwise_distances as pasd



X_train, y_train = load_svmlight_file("./data/dataset_1/train")
x = Util.gen_Circle()

#x = np.array( ( (1,2,3,4,5), (-1,3,-5,8,3) ), np.double).transpose()

numberOfSamples = X_train.shape[0]
numberOfFeatures = X_train.shape[1]

#numberOfSamples = X_train.shape[0]
#numberOfFeatures = X_train.shape[1]

print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))


finalData, vec = akfa(X_train,4,0.0,4,3)

#pl.figure(0)
#pl.title("Original Data")
#pl.plot(x[:,0], x[:,1], 'ro',color='blue')
#pl.plot(x[0][range(30)], x[1][range(30)], 'ro')
#pl.plot(x[0][range(30,90)], x[1][range(30,90)], 'ro',color='blue')
#pl.plot(x[0][range(90,150)], x[1][range(90,150)], 'ro',color='green')




#pl.figure(1)
#pl.title("Data after projecting")
#pl.plot(finalData[:,0].todense(), finalData[:,1].todense(), 'ro',color='blue')
#pl.plot(finalData[0][range(30)], finalData[1][range(30)], 'ro')
#pl.plot(finalData[0][range(30,90)], finalData[1][range(30,90)], 'ro',color='blue')
#pl.plot(finalData[0][range(90,150)], finalData[1][range(90,150)], 'ro',color='green')

#pl.show()
exit()
'''
# testing for csr_matrix akfa here
#===============================================================================
tmp = np.array( ( (1,2,3,4,5), (-1,3,-5,8,3) ), np.double).transpose()

print(tmp)
x  = cdist(tmp,tmp, 'euclidean')
print squareform(cdist(tmp,tmp, 'euclidean'))
print pdist(tmp, 'euclidean')
print x
print x[2,0:5]
exit()
x = Util.gen_Circle()


#K = exp(-squareform(pdist(dataset.transpose(), 'euclidean')) / (2*sigma*sigma))
#K = exp(-cdist(tmp,tmp, 'euclidean') / (2*sigma*sigma))


exit()

pl.figure(0)
pl.title("Original Data")
#pl.plot(x[0,:], x[1,:], 'ro',color='blue')
pl.plot(x[0][range(30)], x[1][range(30)], 'ro')
pl.plot(x[0][range(30,90)], x[1][range(30,90)], 'ro',color='blue')
pl.plot(x[0][range(90,150)], x[1][range(90,150)], 'ro',color='green')



pl.figure(1)
pl.title("Data after projecting")
pl.plot(finalData[0][range(30)], finalData[1][range(30)], 'ro')
pl.plot(finalData[0][range(30,90)], finalData[1][range(30,90)], 'ro',color='blue')
pl.plot(finalData[0][range(90,150)], finalData[1][range(90,150)], 'ro',color='green')

pl.show()

exit()
#idxVectors = akfa(X_train[:,10000][:])
#print(isinstance(x,sparse.csr.csr_matrix))
#print(isinstance(x,np.ndarray))
#===============================================================================


X_train, y_train = load_svmlight_file("./data/dataset_1/train")

numberOfSamples = X_train.shape[0]
numberOfFeatures = X_train.shape[1]

print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
timeBef = time.time()
sigma = 4
chunkSize = 4
chunk = numberOfSamples/chunkSize
for i in range(chunkSize):
    for j in range(chunkSize):
        print("Currently at:")
        print(i,j)
        timeIn = time.time()
        na = exp( - pad(X_train[i*chunk:(i+1)*chunk,:], X_train[j*chunk:(j+1)*chunk,:]) / 2*sigma*sigma)
        print("It took %f to compute the first chunk Gram matrix - if it worked at all" % (time.time()-timeIn))
print("It took %f to compute the Gram matrix - if it worked at all" % (time.time()-timeBef))
print(na[0,0])

exit()
timeIn = time.time()
for i in range(100):
    for j in range(100):
            xy = X_train[0,:].data.shape[0] + X_train[1,:].data.shape[0] - np.intersect1d(X_train[0,:].indices, X_train[1,:].indices).shape[0]*2
            erg = math.exp(-math.sqrt(xy)**2/(2*4*4))
            if i == 0 and j == 0:
                print(erg)
print("It took %f to compute the first chunk Gram matrix - if it worked at all" % (time.time()-timeIn))
print()

print("The Kernel value is:")
print(math.exp(-np.linalg.norm(X_train[0,:].todense() - X_train[1,:].todense(),2)**2/(2*4*4)))
timeIn = time.time()
a = csr_matrix(X_train[0:99,:])
tmp = exp(-cdist(a.todense(),a.todense(), 'euclidean') / (2*4*4))
print(tmp[0,0])
print("It took %f to compute the first chunk Gram matrix - if it worked at all" % (time.time()-timeIn))
exit()

files = hpy.File("akfaData","w")
sparseMatrix = files.create_dataset("sparseMat", (numberOfSamples,numberOfFeatures) , dtype=np.byte)
kernelMatrix = files.create_dataset("kernelMat", (numberOfSamples,numberOfSamples) , dtype=np.float32)

#set = file.create_dataset("sparsedata", data=X_train.todense())
#dset = file.create_dataset("sparsedata", data=exp(squareform(pdist(X_train, 'euclidean'))))
print("let the storing begin")
sigma = 4
#timeNow = time.time()
#for i in range(numberOfSamples):
    #sparseMatrix[i,:] = X_train[i,:].todense()
    #if ( i % 1000 == 0):
        #print(i/1000)
#print("Done!")
#print("It took %f to store the data set on the file system" % (time.time()-timeNow))
print("Creating Kernel Matrix")
timeNow = time.time()
for i in range(0,numberOfSamples,100):
    for j in range(0,numberOfSamples,100):
        timeIn = time.time()
        #kernelMatrix[i:(i+10000),j:(j+10000)] = exp(-cdist(sparseMatrix[i:(i+10000),:],sparseMatrix[j:(i+10000),:], 'euclidean') / (2*sigma*sigma))
        tmp = exp(-cdist(X_train[i:(i+99),:],X_train[j:(j+99),:], 'euclidean') / (2*sigma*sigma))
        print("It took %f to compute the first chunk Gram matrix - if it worked at all" % (time.time()-timeIn))
print("It took %f to compute the Gram matrix - if it worked at all" % (time.time()-timeNow))
print("wait a minute, I am here, so it did work Oo")
print("WUUUUUUUSA")
#rgk = feature.akfa(sparseMatrix[i:(i+10000),:], kernelMatrix, 2, 0.1,4)
exit()
'''
X_train, y_train = load_svmlight_file("./data/dataset_1/train")
numberOfSamples = X_train.shape[0]
numberOfFeatures = X_train.shape[1]

#print(type(X_train))
#print(isinstance(X_train,sparse.csr_matrix))
X_test, y_test = load_svmlight_file("./data/dataset_3/test",n_features=numberOfFeatures)
#X_val, y_val = load_svmlight_file("./data/dataset_3/validate",n_features=X_train.shape[1])

print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))


#print(" ----- ")
#print(" Now choosing the first 100 vectors for testing")
#print(newSet.todense())


train = True


# Run classifier
if train:
    print("Start training")
    # ----------------------------------------------------
    # first, specify classifier
    classifier = svm.SVC(C=1.0,kernel='rbf', probability=True, tol=0.1)
    print("Classifier prepared")
    timeCla = time.time()
    classifier = classifier.fit(X_train, y_train)
    print("Fitting the classifier to data successful in %f" % (time.time()-timeCla))
    # now we can predict
    probas = classifier.predict_proba(X_test)
    print(classifier.get_params())

    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test,probas[:,1])
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
