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


X_train, y_train = load_svmlight_file("./data/dataset_3/train")

X_test, y_test = load_svmlight_file("./data/dataset_3/test",n_features=X_train.shape[1])
#X_val, y_val = load_svmlight_file("./data/dataset_3/validate")



train = True
#clf.predict(X_train)
# Run classifier
if train:
    classifier = svm.SVC(C=1.0,kernel='rbf', probability=True, tol=0.1)
    classifier = classifier.fit(X_train, y_train)
    #finalData, components = akfa(np.array(X_test),5,0.1,'gauss')
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
