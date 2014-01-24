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
import gc
import Util



# testing for csr_matrix akfa here
#===============================================================================
tmp = np.array( ( (1,2,3,4,5), (-1,3,-5,8,3) ), np.double)
x = Util.gen_Circle()


pl.figure(0)
pl.title("Original Data")
pl.plot(x[0][range(30)], x[1][range(30)], 'ro')
pl.plot(x[0][range(30,90)], x[1][range(30,90)], 'ro',color='blue')
pl.plot(x[0][range(90,150)], x[1][range(90,150)], 'ro',color='green')
print(akfa(tmp))

exit()
finalData, comps = akfa(x)
print(comps)
np.savetxt("CircMatrix1.txt", x, fmt='%.18e', delimiter=',', newline='\n')
np.savetxt("CircMatrixResults1.txt", comps, fmt='%.18e', delimiter=',', newline='\n')
pl.figure(1)
pl.title("Data after projecting")
pl.plot(finalData[0][range(30)], finalData[1][range(30)], 'ro')
pl.plot(finalData[0][range(30,90)], finalData[1][range(30,90)], 'ro',color='blue')
pl.plot(finalData[0][range(90,150)], finalData[1][range(90,150)], 'ro',color='green')

pl.show()


#idxVectors = akfa(X_train[:,10000][:])
print(isinstance(x,sparse.csr.csr_matrix))
print(isinstance(x,np.ndarray))
#===============================================================================

X_train, y_train = load_svmlight_file("./data/dataset_1/train")
numberOfSamples = X_train.shape[0]
numberOfFeatures = X_train.shape[1]

X_test, y_test = load_svmlight_file("./data/dataset_3/test",n_features=X_train.shape[1])
X_val, y_val = load_svmlight_file("./data/dataset_3/validate",n_features=X_train.shape[1])

#print("The read file contains %d samples points and  %d features " % (numberOfSamples, numberOfFeatures))
#newSet = X_train[:,100][:]
#print(" ----- ")
#print(" Now choosing the first 100 vectors for testing")
#print(newSet.todense())


train = False


# Run classifier
if train:
    finalData, idxVectors = akfa(X_train)
    # ----------------------------------------------------
    # first, specify classifier
    classifier = svm.SVC(C=1.0,kernel='rbf', probability=True, tol=0.1)
    classifier = classifier.fit(X_train, y_train)
    # now we can predict
    #probas = classifier.predict_proba(X_test)
    print(classifier.get_params())

    
    # Compute ROC curve and area the curve
    #fpr, tpr, thresholds = roc_curve(y_test,probas[:,1])
    #roc_auc = auc(fpr, tpr)
    #print "Area under the ROC curve : %f" % roc_auc
    
    # Plot ROC curve
    pl.clf()
    #pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
