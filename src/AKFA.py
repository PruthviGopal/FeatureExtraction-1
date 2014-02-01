#!/usr/bin/python
'''
Created on 12.03.2013

This file supports a module for doing ACCELERATED KERNEL FEATURE ANALYSIS on a given matrix

@author: christoph
'''
# ---------- Imports ----------#
import numpy as np
from pylab import *
import time
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy import exp
# -------------------- Utility Functions -------------------------#

'''
Used to compute the gaussian kernel of a given matrix
'''
def gaussian(vec_one,vec_two, sigma=4):
    return math.exp(-np.linalg.norm(vec_one.todense() - vec_two.todense(),2)**2/(2*sigma*sigma))

def sparseNorm2(data):
    value = 0.0
    for i in xrange(data.shape[0]):
        value += data[i]*data[i]
    return sqrt(value)

'''
Project components onto new data
'''
def projectKernelComp(dataset,comp,numberOfDataPoints,numberOfFeatures):

    fData = sparse.dok_matrix( (numberOfFeatures,dataset.shape[1]),dtype = np.double)
    print(" ... beginning to project components onto data set")
    #now, we must project the chosen components onto the dataset
    for x in range(numberOfDataPoints):
        #We have now chosen a point in our dataset which is to be adapted
        for index in range(numberOfFeatures):
            tmp = 0
            for i in range(numberOfDataPoints):
                tmp += comp[index,i]*gaussian(dataset[:,i],dataset[:,x])
            fData[index,x] = tmp
    print("....... projecting finished!")
    return fData.tocsr()



# -------------------- Code -------------------------#

def akfa(dataset, featureNumber=2, delta = 0.0, sigma=4,):
    #dataset = sparse.dok_matrix(dataset.transpose(),dtype=np.double)
    print("___________")
    print("Setting up for Feature Extraction via Accelerated Kernel Feature Analysis")
    print(" ... %d features are to be extracted!" %(featureNumber))
    print("Data set begin loaded...")
    print(".....")
    print(".....")
    # first off, we need to check if the given data set is a numpy array or a sparse matrix
    if( isinstance(dataset,np.ndarray)):
        print("Data set is not a sparse matrix --> reconfiguring into sparse matrix")
        dataset = sparse.dok_matrix(dataset,dtype=np.double)
        print(".....")
        print("Conversion to sparse matrix successfull")
    #elif( (not isinstance(dataset,sparse.csr_matrix)) or (not isinstance(dataset,sparse.csc.csc_matrix) )):
        #raise Exception("The data set is not in the right format")
    print("Loading of data set is completed")
    print(".....")
    # Here we need to transform the matrix, so that we can access the vectors easier
    #dataset = dataset.transpose()
    n = dataset.shape[0]
    # Now we can compute the Gram Matrix
    print("The loaded file contains %d samples points and %d dimensions " % (n, dataset.shape[0]))
    print(".....")
    K = sparse.dok_matrix( (n,n),dtype = np.double)
    print(".....")
    print("Creating new Sparse Matrix for holding the Gram Matrix")
    print("For a large data set, this may take a while ...")
    #timeMatrix = time.time()
    #for i in range(n):
        #for j in range(n):
            #K[i,j] = gaussian(dataset[:,i], dataset[:,j],sigma)
        #print("Done for row [%d]"%(i))
    #print("It took %f seconds to build the Gram matrix" %(time.time()-timeMatrix))
    #K = K.tocsr()
    dist = pdist(dataset.todense(), 'euclidean')
    pairwise_dists = squareform(dist)
    K = exp(pairwise_dists / sigma**2)
    print(".....")
    print("Done computing the Gram Matrix")
    print("......")
    print("............")
    print("Will continue with extracting features now!")
    #center the matrix in feature space
    # LEAVE THAT OUT FOR NOW
    #N = np.multiply(1./dataset.shape[0],np.ones( (datase(dataset.getrow(i)t.shape[0],dataset.shape[0]) , dtype = np.double) ) 
    #K = np.subtract(dataset,np.dot(N,dataset))
    #K = np.subtract(K,np.dot(dataset,N))
    #K = np.add(K,np.dot(N, np.dot(dataset,N)))
    # Stop the time
    timeBefore = time.time()
    # Now we need variables to hold the extracted components
    idx = 0
    idxVectors = np.zeros( (featureNumber,n),dtype = np.double)

    # Now we can start to extract features
    for i in range(featureNumber):
        print("___________________")
        print("Starting extracting of feature %d" % (i+1))
        # Extract i-th feature using (11)
        maxValue = 0
        maxIn = 0
        print("....")
        for j in range(n):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if K[j,j] <= delta:
                continue
            #sumOf = (1/(n*K[j,j])) * K.getcol(j).multiply(K.getcol(j)).sum()
            sumOf = (1/(n*K[j,j])) * np.multiply(K[j,:],K[j,:]).sum()
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        # NEED TO CHECK FOR ZERO VALUES IN DATA
        print("........")
        idxVectors[i,:] =  K[maxIn,:] / sparseNorm2(K[maxIn,:])
        idx = maxIn
        print("__Feature found!")
        if i == featureNumber-1:
            continue
        # Now we must use equation (10) to update the Kernel Matrix K       
        K_new = sparse.dok_matrix( (n,n),dtype = np.double)
        print("Updating Gram Matrix!")
        for j in range(n):
            for k in range(n):
                K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
        K = K_new.tocsr()
        #K.eliminate_zeros()
        print("Update successfull - continue!")
        
    tmpTime = time.time()
    print("__________")
    print("It took that many seconds to compute the data with AKFA: %f" % (tmpTime-timeBefore))
    print("__________")
    finalData = projectKernelComp(dataset, idxVectors,n,featureNumber)
    print("__________")
    print("It took that many seconds to project the components onto the data: %f" % (time.time()-tmpTime))
    print("__________")
    return finalData.tocsr(), idxVectors


    