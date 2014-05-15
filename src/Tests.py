'''
Created on 08.01.2014

@author: b4mbi
'''
from sklearn.datasets import load_svmlight_file, make_blobs, make_circles
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import pylab as pl
from AKFA import akfa, projectKernelComp, buildGramMatrix, kpca,oldKernelMatrix
import numpy as np
from scipy import sparse
import Util
import time
from scipy.sparse import csr_matrix
import cPickle
import cProfile
from matplotlib.pyplot import *
from pylab import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot
import pylab
import random


fig = pl.figure()
ax = fig.add_subplot(111,projection='3d')

pl.title("3D example data")
data, labels = make_blobs(500, 3, centers=4, cluster_std=0.75 , shuffle=True)
data = data.transpose()
reds = labels == 0
blues = labels == 1
greens = labels == 2
blacks = labels == 3
ax.scatter(data[0,reds], data[1,reds], data[2,reds], c='r', marker='o')
ax.scatter(data[0,blues], data[1,blues], data[2,blues], c='b', marker='o')
ax.scatter(data[0,greens], data[1,greens], data[2,greens], c='g', marker='o')
ax.scatter(data[0,blacks], data[1,blacks], data[2,blacks], c='k', marker='o')

pyplot.show()


exit()
data, labels = make_blobs(3000, 100, centers=6, cluster_std=0.4 , shuffle=True)

K = buildGramMatrix(data, 3000, 5, 4)
np.savetxt("matrix.txt", data)
np.savetxt("kernelMatrix.txt", K)
exit()
for i in range(5,61,5):
    print("Extracting %d features" % i)
    
    print("____________________________________________________________________________________________________________________________________________________")
    print("Starting of extracting features: %d" % i)
    print("____________________________________________________________________________________________________________________________________________________")
    print("AKFA with delta = 0.0")
    print("_____________________________________")
    cProfile.run('akfa(dataset=csr_matrix(data),n_features=i,delta=0.0,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)')
    print("AKFA with delta = 0.3")
    print("_____________________________________")
    cProfile.run('akfa(dataset=csr_matrix(data),n_features=i,delta=0.3,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)')
    print("KPCA")
    print("_____________________________________")
    cProfile.run('kpca(dataset=csr_matrix(data), n_features=i, sigma=4, chunkSize = 5,isMatrixGiven=True, K=K)')
exit()













pl.figure(0)
pl.title("Correlation between Number of Extracted Features and AUC")
x = [1000,1500,2000,2500,3000,5000,6000, 7500,10000,15000]
y = [0.40,0.39,0.43,0.45,0.56,0.67,0.70, 0.71, 0.78,0.80]
p1, = pl.plot(x,y, "g-o")
pl.ylim([0.0, 1.0])
pl.xlim([1000, 15000])
pl.xlabel("Number of Extracted Features")
pl.ylabel("AUC")
l1 = pl.legend([p1], ["Feature to Roc"], loc=2)
pl.show()




exit()
o = 3500
# ./test 3500 100 20 "matrix.txt" "results_C.txt" "kernelMatrix.txt" 0.0
data, labels = make_blobs(o, 100, centers=6, cluster_std=0.4 , shuffle=True)
np.save("matrix.txt", data)
K = buildGramMatrix(data, o, 5, 4)
np.save("kernelMatrix.txt", K)
exit()
for i in range(500,10000,500):
    f = 20
    print("Size of data set: %d x 100" % i)
    data, labels = make_blobs(i, 100, centers=6, cluster_std=0.4 , shuffle=True)
    K = buildGramMatrix(data, i, 5, 4)
    print("____________________________________________________________________________________________________________________________________________________")
    print("Starting of extracting features: %d" % f)
    print("____________________________________________________________________________________________________________________________________________________")
    print("AKFA with delta = 0.0")
    print("_____________________________________")
    cProfile.run('akfa(dataset=csr_matrix(data),n_features=f,delta=0.0,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)')
    print("AKFA with delta = 0.3")
    print("_____________________________________")
    cProfile.run('akfa(dataset=csr_matrix(data),n_features=f,delta=0.3,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)')
    print("KPCA")
    print("_____________________________________")
    cProfile.run('kpca(dataset=csr_matrix(data), n_features=f, sigma=4, chunkSize = 5,isMatrixGiven=True, K=K)')
exit()




pl.figure(0)
pl.title("Correlation between Number of Extracted Features and AUC")
x = [1000,1500,2000,2500,3000,5000,10000,15000]
y = [0.40,0.39,0.43,0.45,0.56,0.67,0.78,0.80]
p1, = pl.plot(x,y, "g-o")
pl.ylim([0.0, 1.0])
pl.xlim([1000, 15000])
l1 = pl.legend([p1], ["Feature to Roc"], loc=2)

pl.figure(1)
pl.title("Comparrision of Computation Time for AKFA, ACKFA and KPCA")
pl.xlabel('Number of Extracted Features')
pl.ylabel('Time t')


x = range(5,31,5)
akfa = [1.698 ,3.866 ,5.948 ,7.656 ,10.253 ,12.543]
ackfa = [1.768 ,3.692 ,5.486 ,7.299 ,8.882 ,9.976]
kpca = [3.185, 16.591, 21.636, 28.647, 31.149, 29.980]
c_akfa = [0.29, 0.54, 0.77, 1.01, 1.25, 1.5]
c_ackfa = [0.29, 0.53, 0.77, 1.02, 1.25, 1.53]
p1, = pl.plot(x,akfa, "r-o")
p2, = pl.plot(x,ackfa, "b-o")
p3, = pl.plot(x,kpca, "g-o")
p4, = pl.plot(x,c_akfa, "k-o")
p5, = pl.plot(x,c_ackfa, "y-o")
pl.yscale('log')
l1 = pl.legend([p1,p2,p3,p4,p5], ["AKFA","ACKFA","KPCA","C_AKFA","C_ACKFA"], loc=2)

pl.show()



exit()




exit()


#np.savetxt("Blobmatrix.txt",data)
#print("Saving successfull")
#cProfile.run('akfa(dataset=csr_matrix(data),n_features=20,delta=0.3,sigma=4,chunkSize=5,isMatrixGiven=False,K=None)')
#exit()



small = 400
med = 800
big = 800
#x = Util.gen_Circle(small = small, med = med, big = big)
x = np.loadtxt("CircMatrix1.txt")   
comps = np.loadtxt("Result.txt")
numberOfSamples = x.shape[0]
numberOfFeatures = x.shape[1]
print("The read file contains %d samples points and  %d dimensions " % (numberOfSamples, numberOfFeatures))
print("_____________________________________")
print("Number of Sample Points: %d" % numberOfSamples)
print("_____________________________________")



#cProfile.run('akfa(dataset=csr_matrix(x),n_features=2,delta=0.5,sigma=4,chunkSize=5,isMatrixGiven=False,K=None)')

y = projectKernelComp(x, comps, 4)

train = projectKernelComp(X_train, comps[0:10000,:], 4)

pl.plot(x[range(small),0],x[range(small),1], "ro")
pl.plot(x[range(small,small+med),0],x[range(small,small+med),1], "bo")
pl.plot(x[range(small+med,small+med+big),0],x[range(small+med,small+med+big),1], "go")

x1 = y[:,0].todense()
y1 = y[:,1].todense()
pl.figure(1)
pl.plot(x1[range(small)],y1[range(small)], "ro")
pl.plot(x1[range(small,small+med)],y1[range(small,small+med)], "bo")
pl.plot(x1[range(small+med,small+med+big)],y1[range(small+med,small+med+big)], "go")
pl.show()
exit()

##X_train, y_train = load_svmlight_file("./data/dataset_5/train")
#X_train = X_train[0:10000,:]
#y_train = y_train[0:10000]
#numberOfSamples = X_train.shape[0]
#numberOfFeatures = X_train.shape[1]

#for i in range(0,700,100):
#
    #small = 0+i
    #med = 200+i
    #big = 200+i
#    
    #numberOfSamples = small + big + med
    #print("_____________________________________")
    #print("Number of Sample Points: %d" % numberOfSamples)
#    p#rint("_____________________________________")
    #x = Util.gen_Circle(small = small, med = med, big = big)
    #cProfile.run('buildGramMatrix(x, numberOfSamples, 5, 4)')
    #cProfile.run('oldKernelMatrix(x, numberOfSamples, 5, 4)')

exit()

X_train, y_train = load_svmlight_file("./data/dataset_3/train")
X_train = X_train[0:5000,:]
y_train = y_train[0:5000]
numberOfSamples = X_train.shape[0]
numberOfFeatures = X_train.shape[1]

X_test, y_test = load_svmlight_file("./data/dataset_5/test",n_features=numberOfFeatures)
X_test = X_test[0:10000,:]
y_test = y_test[0:10000]
numberOfTestSamples = X_test.shape[0]

#comps = akfa(X_train, 5000, 0.0, 4, 5, False, None)
#np.save('./data/dataset_4/components.npy', comps)
comps = np.load('./data/dataset_5/components.npy')
print(comps.shape)
# 1000 features
train = projectKernelComp(X_train, comps[0:10000,:], 4)
test = projectKernelComp(X_test, comps[0:10000,:], 4)
cPickle.dump(train, open("./data/dataset_5/projData/train10000comps.pickle", "w"))
cPickle.dump(test, open("./data/dataset_5/projData/test10000comps.pickle", "w"))
# 1250 features
train = projectKernelComp(X_train[0:5000,:], comps[0:5000,:], 4)
test = projectKernelComp(X_test[0:5000,:], comps[0:5000,:], 4)
cPickle.dump(train, open("./data/dataset_5/projData/train5000comps.pickle", "w"))
cPickle.dump(test, open("./data/dataset_5/projData/test5000comps.pickle", "w"))
# 1500 features
#train = projectKernelComp(X_train, comps[0:1500,:], 4)
#test = projectKernelComp(X_test, comps[0:1500,:], 4)
#cPickle.dump(train, open("./data/dataset_4/projData/train1500comps.pickle", "w"))
#cPickle.dump(test, open("./data/dataset_4/projData/test1500comps.pickle", "w"))
# 1750 features
#train = projectKernelComp(X_train, comps[0:1750,:], 4)
#test = projectKernelComp(X_test, comps[0:1750,:], 4)
#cPickle.dump(train, open("./data/dataset_4/projData/train1750comps.pickle", "w"))
#cPickle.dump(test, open("./data/dataset_4/projData/test1750comps.pickle", "w"))
# 2000 features
#train = projectKernelComp(X_train, comps[0:2000,:], 4)
#test = projectKernelComp(X_test, comps[0:2000,:], 4)
#cPickle.dump(train, open("./data/dataset_4/projData/train2000comps.pickle", "w"))
#cPickle.dump(test, open("./data/dataset_4/projData/test2000comps.pickle", "w"))




exit()
y = akfa(dataset=csr_matrix(x),n_features=2,delta=0.5,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)

test = False


if test:
    
    # --- WORKING TEST WITH SMALL CIRCLE
    
    small = 100
    med = 300
    big = 300
    
    
    x = Util.gen_Circle(small = small, med = med, big = big)
    pl.figure(0)
    pl.plot(x[range(small),0],x[range(small),1], "ro")
    pl.plot(x[range(small,small+med),0],x[range(small,small+med),1], "bo")
    pl.plot(x[range(small+med,small+med+big),0],x[range(small+med,small+med+big),1], "go")
    numberOfSamples = x.shape[0]
    numberOfFeatures = x.shape[1]
    
    K = buildGramMatrix(x, numberOfSamples, 5, 4)
    print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
    y = akfa(dataset=csr_matrix(x),n_features=2,delta=0.5,sigma=4,chunkSize=5,isMatrixGiven=True,K=K)
    exit()
    #x1 = data[:,0].todense()
    #y1 = data[:,1].todense()
    pl.figure(1)
    #pl.plot(x1[range(small)],y1[range(small)], "ro")
    #pl.plot(x1[range(small,small+med)],y1[range(small,small+med)], "bo")
    #pl.plot(x1[range(small+med,small+med+big)],y1[range(small+med,small+med+big)], "go")
    pl.show()
    
    exit()
    
if not test:  
    

    X_train, y_train = load_svmlight_file("./data/dataset_3/train")
    X_train = X_train[0:5000,:]
    y_train = y_train[0:5000]

    

    numberOfSamples = X_train.shape[0]
    numberOfFeatures = X_train.shape[1]
    
    

   

    X_test, y_test = load_svmlight_file("./data/dataset_3/test",n_features=numberOfFeatures)
    X_test = X_test[0:5000,:]
    y_test = y_test[0:5000]

    numberOfTestSamples = X_test.shape[0]
    
    
    
    
    
    
    

    
   
    
    #print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
    
    
    
    #print("Start comparring AKFA and KPCA and AKCFA")
    
    #for i in range(0,5000,500):
        #X_trainIn = X_train[0:(i+500),:]
        #y_trainIn = y_train[0:(i+500)]
        #akfaRes = akfa(X_trainIn, 100, 0.0, 4, 5, False, None)
        #kpcaVec, kpcaEig = kpca(X_trainIn,100, 4, 5, False, None)
        #akfaCutRes= akfa(X_trainIn, 100, 0.4, 4, 5, False, None)
    
    
    #exit()
    
    
    
    

   
    

    

    
    train = True

    comps = np.load('./data/dataset_3/components.npy')
    print(comps.shape)
    if ( comps.shape[0] != 2000):
        print("ERROR")
        print(comps.shape)
        exit()
    print("Successfully found components in file")
    
    train = np.fromfile('./data/dataset_3/dataTrain2000comps.npy')
    test = np.load('./data/dataset_3/dataTest2000comps.npy')
    
    
    print(train.shape)
    
    exit()
    
    #print("Successfully found components in file")

    
    
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
    
        #pl.show()
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test,probas[:,1])
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve for NORMAL LEARNING: %f" % roc_auc
        pl.figure(0)
        pl.title("ROC for normal learning")
        # Plot ROC curve
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        
        print("________")
        print("OK, now do the same thing for projected data!")
        print("________")
        
        # _______ TRAINING FOR REPORJECTED DATA ___________________________________________________________________________ features = 250
        
        
        classifier = svm.SVC(C=1.0,kernel='rbf', probability=True, tol=0.1)
        print("Classifier prepared")
        timeCla = time.time()
        classifier = classifier.fit(train, y_train)
        print("Fitting the classifier to data successful in %f" % (time.time()-timeCla))
        # now we can predict
        probas = classifier.predict_proba(test)
        print(classifier.get_params())
        #pl.show()
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test,probas[:,1])
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve with REDUCED DATA SET: %f" % roc_auc
        pl.figure(1)
        pl.title("ROC for projected learning with 250 features")
        # Plot ROC curve
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        
        
        pl.savefig('res.png')
        pl.show()
