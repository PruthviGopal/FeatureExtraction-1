#!/usr/bin/python
'''
Created on 12.03.2013

This file supports a module for doing ACCELERATED KERNEL FEATURE ANALYSIS on a given matrix

@author: christoph
'''
# ---------- Imports ----------#
import numpy as np
import math as ma
import time
from scipy import sparse
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import exp
from scipy import spatial
from AKFA import projectKernelComp
cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t
# -------------------- Utility Functions -------------------------#

'''
Used to compute the gaussian kernel of a given matrix
'''
cdef double gaussian(np.ndarray vec_one,np.ndarray vec_two, sigma=4):
    #return exp(-np.linalg.norm(vec_one - vec_two,2)**2/(2*sigma*sigma))
    return exp(-spatial.distance.euclidean(vec_one,vec_two)**2/(2*sigma*sigma))

cdef double sparseNorm2(np.ndarray data):
    value = 0.0
    for i in xrange(data.shape[0]):
        value += data[i]*data[i]
    return ma.sqrt(value)

'''
Project components onto new data
'''
def projectKernelComp(dataset,np.ndarray comp,int numberOfDataPoints,int numberOfFeatures):
    cdef double tmp = 0
    cdef np.ndarray fData = np.zeros( (numberOfDataPoints,numberOfFeatures),dtype = np.double)
    tmpTime = time.time()
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
    print("..it took %f seconds to project the data"%(time.time()-tmpTime))
    return fData



# -------------------- Code -------------------------#

def akfa(dataset, int featureNumber=2, double delta = 0.0, int sigma=4,):
    
    nparray = isinstance(dataset,np.ndarray)
    
    '''
    First off, we need to specify our variables for this function
    '''
    # The number of Data Points in the data set
    cdef int n = dataset.shape[0]
    cdef int dim = dataset.shape[1]
    # A Matrix for holding the Gram Matrix
    cdef np.ndarray K = np.zeros( (n,1),dtype = np.double)
    # A temporary Matrix for holding the Gram Matrix during the update
    #cdef np.ndarray K_new = np.zeros( (n,n),dtype = np.double)
    # Index for sotring the chosen vector in the Gram Matrix
    cdef np.ndarray idx = np.arange(featureNumber)
    # Array for holding the chosen components
    cdef np.ndarray idxVectors = np.zeros( (featureNumber,n),dtype = np.double)
    # Auxilirary variables
    cdef double maxValue = 0
    cdef int maxIn = 0
    cdef double sumOf = 0
    cdef double up = 0
    print("OH MY GOD I AM A CYTHON FUNCTION")
    #dataset = sparse.dok_matrix(dataset.transpose(),dtype=np.double)
    print("___________")
    print("Setting up for Feature Extraction via Accelerated Kernel Feature Analysis")
    print(" ... %d features are to be extracted!" %(featureNumber))
    print("Data set begin loaded...")
    print(".....")
    print(".....")
    print("Loading of data set is completed")
    print(".....")
    # Here we need to transform the matrix, so that we can access the vectors easier
    #dataset = dataset.transpose()
    
    # Now we can compute the Gram Matrix
    print("The loaded file contains %d samples points and %d dimensions " % ( n,dim))
    print(".....")
    
    print(".....")
    print("NOT Creating the Gram Matrix of size %dx%d"%(n,n))
    print("For a large data set, this may take a while ...")
    
    #timeMatrix = time.time()
    #for i in range(n):
        #for j in range(n):
            #K[i,j] = gaussian(dataset[:,i], dataset[:,j],sigma)
        #print("Done for row [%d]"%(i))
    #print("It took %f seconds to build the Gram matrix" %(time.time()-timeMatrix))

    #pairwise_dists = squareform(pdist(dataset.transpose(), 'euclidean'))
    #K = exp(pairwise_dists**2 / (2*sigma*sigma))
    #K = exp(-squareform(pdist(dataset.transpose(), 'euclidean')) / (2*sigma*sigma))

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
            timeVec = time.time()
            for k in range(n):
                if nparray:
                    #tmp = cdist(xa.transpose(),dataset.transpose())
                    K[k,0] = gaussian(dataset[j,:], dataset[k,:],sigma)
                else:
                    K[k,0] = gaussian(dataset[j,:].todense(), dataset[k,:].todense(),sigma)
            if i == 0:
                print("Successfully build vector %d in %f"%(j,time.time()-timeVec))
            for f in range(i):
                for k in range(n):
                    #K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
                    up = (idxVectors[f,j]*idxVectors[f,k])/idxVectors[f,idx[f]]
                    K[k,0] = K[k,0] - up
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if K[j,0] <= delta:
                continue
            #sumOf = (1/(n*K[j,j])) * K.getcol(j).multiply(K.getcol(j)).sum()
            sumOf = (1/(n*K[j,0])) * np.multiply(K[:,0],K[:,0]).sum()
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        print("__Feature found!")
        print("........")
        for k in range(n):
            if not nparray:
                K[k,0] = gaussian(dataset[maxIn,:].todense(), dataset[k,:].todense(),sigma)
            else:
                K[k,0] = gaussian(dataset[maxIn,:], dataset[k,:],sigma)
        idxVectors[i,:] =  K[:,0] #/ sparseNorm2(K[:,0])
        idx[i] = maxIn
        print("Feature found and successfully stored!")
        if i == featureNumber-1:
            continue
        # Now we must use equation (10) to update the Kernel Matrix K       
        #K_new = np.zeros( (n,n),dtype = np.double)
        #print("Updating Gram Matrix!")
        #for j in range(n):
            #for k in range(n):
                #K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
                #[j,k] = [j,k] - 
        #K = K_new
        #K.eliminate_zeros()
        print("_continue!")
    '''
    NEED TO PUT THE CALCULATION OF THE NORM INTO THE FEATURES HERE
    '''
    for i in range(featureNumber):
        idxVectors[i,:] = idxVectors[i,:] / sparseNorm2(idxVectors[i,:])
    tmpTime = time.time()
    print("__________")
    print("It took that %f seconds to compute the data with AKFA" % (tmpTime-timeBefore))
    print("__________")
    return idxVectors
    return projectKernelComp(dataset,idxVectors,n,featureNumber),idxVectors


    