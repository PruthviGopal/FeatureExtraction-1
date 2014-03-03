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
import h5py as hpy
from sklearn.metrics.pairwise import pairwise_distances as padis
# -------------------- Utility Functions -------------------------#

'''
Used to compute the gaussian kernel of a given matrix
'''
def gaussian(vec_one,vec_two, sigma=4):
    return math.exp(-np.linalg.norm(vec_one.todense() - vec_two.todense(),2)**2/(2*sigma*sigma))


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

def akfa(dataset, featureNumber=2, delta = 0.0, sigma=4, chunkSize = 4):
    print("___________")
    print("Setting up for Feature Extraction via Accelerated Kernel Feature Analysis")
    print(" ... %d features are to be extracted!" %(featureNumber))
    print("Data set begin loaded...")
    print(".....")
    print(".....")
    print("Loading of data set is completed")
    print(".....")
    n = dataset.shape[0]
    numberOfFeatures = dataset.shape[1]
    # Now we can compute the Gram Matrix
    print("The loaded file contains %d samples points and %d dimensions " % (n,numberOfFeatures ))
    print(".....")
    # Now we need to create a file that holds the Gram Matrix
    files = hpy.File("akfaData","w")
    kernelMatrix = files.create_dataset("kernelMat", (n,n) , dtype=np.float32)
    print(".....")
    print("Creating new Sparse Matrix for holding the Gram Matrix")
    print("For a large data set, this may take a while ...")
    chunk = n/chunkSize
    print("Chunking data set into %d different sents" % (chunkSize))
    print(".... now starting to compute %d x %d blocks of Gram Matrix" % (chunk,chunk) )
    timeBef = time.time()
    for i in range(chunkSize):
        for j in range(chunkSize):
            kernelMatrix[i*chunk:(i+1)*chunk,j*chunk:(j+1)*chunk] = exp( - padis(dataset[i*chunk:(i+1)*chunk,:], dataset[j*chunk:(j+1)*chunk,:]) / 2*sigma*sigma)
            #tmp = exp( - padis(dataset[i*chunk:(i+1)*chunk,:], dataset[j*chunk:(j+1)*chunk,:]) / 2*sigma*sigma)
        print("  ... Finished the first %d rows" % ((i+1)*chunk))
    print("It took %f to compute the Gram matrix" % (time.time()-timeBef))
    print(".....")
    print("Therefore: Done computing the Gram Matrix")
    print("WUHU!!")
    print("......")
    print("............")
    print("Will continue with extracting features now!")
    
    # -----------------------------------------------------
    #center the matrix in feature space
    # LEAVE THAT OUT FOR NOW
    #N = np.multiply(1./dataset.shape[0],np.ones( (datase(dataset.getrow(i)t.shape[0],dataset.shape[0]) , dtype = np.double) ) 
    #K = np.subtract(dataset,np.dot(N,dataset))
    #K = np.subtract(K,np.dot(dataset,N))
    #K = np.add(K,np.dot(N, np.dot(dataset,N)))
    # -----------------------------------------------------
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
        timeIn = time.time()
        print("....")
        for j in range(n):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if kernelMatrix[j,j] <= delta:
                continue
            #sumOf = (1/(n*K[j,j])) * K.getcol(j).multiply(K.getcol(j)).sum()
            sumOf = (1/(n*kernelMatrix[j,j])) * np.multiply(kernelMatrix[j,:],kernelMatrix[j,:]).sum()
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        # NEED TO CHECK FOR ZERO VALUES IN DATA
        print("........")
        idxVectors[i,:] =  kernelMatrix[maxIn,:] / np.linalg.norm(kernelMatrix[maxIn,:])
        idx = maxIn
        print("__Feature found!")
        print("_____ ... which took %f " % (time.time()-timeIn))
        if i == featureNumber-1:
            continue
        idxVec = kernelMatrix[:,idx]
        print(idxVec.shape)
        print(idxVec[0])
        print("Now updating the Gram Matrix")
        timeBef = time.time()
        # Now we must use equation (10) to update the Kernel Matrix K
        for i in range(n):
            fak = idxVec[i]/idxVec[idx]
            kernelMatrix[:,i] = np.subtract(kernelMatrix[:,i],idxVec*fak)
            if ( i % 1000 == 0):
                print("For first %d it took %f" % (i,time.time()-timeBef))  
        #K_new = sparse.dok_matrix( (n,n),dtype = np.double)
        #print("Updating Gram Matrix!")
        #for j in range(n):
            #for k in range(n):
                #K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
        #K = K_new.tocsr()
        #K.eliminate_zeros()
        print("Update successfull - in %f " % (time.time()-timeBef))
        print(" ------> continue!")
        
    tmpTime = time.time()
    print("__________")
    print("It took that many seconds to compute the data with AKFA: %f" % (tmpTime-timeBefore))
    print("__________")
    finalData = projectKernelComp(dataset, idxVectors,n,featureNumber)
    print("__________")
    print("It took that many seconds to project the components onto the data: %f" % (time.time()-tmpTime))
    print("__________")
    return finalData.tocsr(), idxVectors


    