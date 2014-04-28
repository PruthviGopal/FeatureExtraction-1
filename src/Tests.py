'''
Created on 08.01.2014

@author: b4mbi
'''
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import pylab as pl
from AKFA import akfa, projectKernelComp, buildGramMatrix, kpca
import numpy as np
from scipy import sparse
import Util
import time
from scipy.sparse import csr_matrix
import cPickle
import cProfile

X_train, y_train = load_svmlight_file("./data/dataset_5/train")
X_train = X_train[0:10000,:]
y_train = y_train[0:10000]

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
