'''
Created on 08.01.2014

@author: b4mbi
'''
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import pylab as pl
from AKFA import akfa, projectKernelComp
import numpy as np
from scipy import sparse
import Util
import time
from scipy.sparse import csr_matrix


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
    print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
    data,y = akfa(csr_matrix(x),2,0.5,4,5)
    
    x1 = data[:,0].todense()
    y1 = data[:,1].todense()
    pl.figure(1)
    pl.plot(x1[range(small)],y1[range(small)], "ro")
    pl.plot(x1[range(small,small+med)],y1[range(small,small+med)], "bo")
    pl.plot(x1[range(small+med,small+med+big)],y1[range(small+med,small+med+big)], "go")
    pl.show()
    
    exit()
    
if not test:  
    
    X_train, y_train = load_svmlight_file("./data/dataset_1/train")
    numberOfSamples = X_train.shape[0]
    numberOfFeatures = X_train.shape[1]
    
    #print(type(X_train))
    #print(isinstance(X_train,sparse.csr_matrix))
    X_test, y_test = load_svmlight_file("./data/dataset_1/test",n_features=numberOfFeatures)
    #X_val, y_val = load_svmlight_file("./data/dataset_3/validate",n_features=X_train.shape[1])
    numberOfTestSamples = X_test.shape[0]
    print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
    
    finalData, comps = akfa(X_train,10,0.5,4,3)
    
    print(".....")
    print("Re-projecting Test Data set")
    
    testData = projectKernelComp(X_test, comps, numberOfTestSamples, 10, 4)
    
    print("Done projecting, start training, damn it!")
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
        # _______ TRAINING FOR REPORJECTED DATA
        
        classifier = svm.SVC(C=1.0,kernel='rbf', probability=True, tol=0.1)
        print("Classifier prepared")
        timeCla = time.time()
        classifier = classifier.fit(finalData, y_train)
        print("Fitting the classifier to data successful in %f" % (time.time()-timeCla))
        # now we can predict
        probas = classifier.predict_proba(testData)
        print(classifier.get_params())
    
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test,probas[:,1])
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve with REDUCED DATA SET: %f" % roc_auc
        pl.figure(0)
        pl.title("ROC for projected learning")
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
