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
def projectKernelComp(dataset,comp,n):
    #matrix = learningset.transpose()
    numberOfDimensions = dataset.shape[1]
    fData = sparse.csr_matrix( np.zeros((dataset.shape[0],dataset.shape[1]),dtype = np.double))
    #now, we must project the chosen components onto the dataset
    for x in range(n):
        #We have now chosen a point in our dataset which is to be adapted
        y = dataset[x,:]
        for index in range(n):
            tmp = 0
            for i in range(numberOfDimensions):
                tmp += comp[index,i]*gaussian(dataset[i,:],y)
            y[0,index] = tmp
        fData[x,:] = tmp
    return fData



# -------------------- Code -------------------------#

def akfa(dataset, featureNumber=2, delta = 0.0, sigma=4,):
    print("___________")
    print("Setting up for Feature Extraction via Accelerated Kernel Feature Analysis")
    print("Data set begin loaded...")
    print(".....")
    print(".....")
    # first off, we need to check if the given data set is a numpy array or a sparse matrix
    if( isinstance(dataset,np.ndarray)):
        print("Data set is not a sparse matrix --> reconfiguring into sparse matrix")
        dataset = sparse.csr_matrix(dataset,dtype=np.double)
        print(".....")
        print("Conversion to sparse matrix successfull")
    elif( not isinstance(dataset,sparse.csr_matrix) or not isinstance(dataset,sparse.csc.csc_matrix) ):
        raise Exception("The data set is not in the right format")
    print("Loading of data set is completed")
    print(".....")
    # Here we need to transform the matrix, so that we can access the vectors easier
    dataset = dataset.transpose()
    n = dataset.shape[0]
    # Now we can compute the Gram Matrix
    print("The loaded file contains %d samples points and %d dimensions " % (n, dataset.shape[1]))
    print(".....")
    K = sparse.csr_matrix( np.zeros((n,n),dtype = np.double))
    print(".....")
    print("Creating new Sparse Matrix for holding the Gram Matrix")
    for i in range(n):
        #tmp_indices = np.array([],dtype = np.double)
        #tmp_indptr = np.array([],dtype = np.double)
        #tmp_data = np.array([],dtype = np.double)
        for j in range(n):
            K[i,j] = gaussian(dataset[i,:], dataset[j,:],sigma)
            #tmp_indices = np.append(tmp_indices,j)
            #tmp_data = np.append(tmp_data,gaussian(dataset.getrow(i), dataset.getrow(j),sigma))
            #K.getrow(i)[j] = gaussian(dataset.getrow(i), dataset.getrow(j),sigma)
            #print("Value is: %d " % (K.getcol(i)[j]))
        #tmp_indptr = np.append(tmp_indptr,i*n)
        #K.data = np.append(K.data,tmp_data)
        #K.indices = np.append(K.indices,tmp_indices)
        #K.indptr = np.append(K.data,tmp_indptr)
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
    idxVectors = np.zeros((featureNumber, n), np.double)
    # Now we can start to extract features
    for i in range(featureNumber):
        print("___________________")
        print("Starting extracting of feature %d" % (i))
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
            sumOf = K.getcol(j).multiply(K.getcol(j)).sum()
            sumOf = (1/(n*K[j,j]) * sumOf)
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        # NEED TO CHECK FOR ZERO VALUES IN DATA
        print("........")
        idxVectors[i] =  K.getcol(maxIn).data / sparseNorm2(K.getcol(maxIn).data)
        idx = maxIn
        print("__Feature found!")
        # Now we must use equation (10) to update the Kernel Matrix K       
        K_new = sparse.csr_matrix( np.zeros((n,n),dtype = np.double))
        print("Updating Gram Matrix!")
        for j in range(n):
            for k in range(n):
                K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
        K = K_new
        print("Update successfull - continue!")
        
    timeAfter = time.time()
    print("__________")
    print("It took that many seconds to compute the data with AKFA: %f" % (timeAfter-timeBefore))
    print("__________")
    finalData = projectKernelComp(dataset, idxVectors,n)
    return finalData, idxVectors


    