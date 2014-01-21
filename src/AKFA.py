#!/usr/bin/python
'''
Created on 12.03.2013

This file supports a module for doing ACCELERATED KERNEL FEATURE ANALYSIS on a given matrix

@author: christoph
'''
# ---------- Imports ----------#
import Util
import numpy as np
from pylab import *
import time

# -------------------- Utility Functions -------------------------#

'''
Used to compute the gaussian kernel of a given matrix
'''
def gaussian(vec_one,vec_two, sigma=4):
    print(vec_one)
    return math.exp(-np.linalg.norm(vec_one - vec_two,2)**2/(2*sigma*sigma))



# -------------------- Code -------------------------#

def akfa(dataset, featureNumber=2, delta = 0.0, sigma=4):
    # Here we need to transform the matrix, so that we can access the vectors easier
    matrix = dataset.transpose()
    # Now we can compute the Gram Matrix
    n = matrix.shape[0]
    K = np.zeros((n,n), float)
    for i in range(n):
        for j in range(n):
            K[i][j] = gaussian(matrix[i], matrix[j],sigma)
    # center the matrix in feature space
    # LEAVE THAT OUT FOR NOW
    #N = np.multiply(1./matrix.shape[0],np.ones( (matrix.shape[0],matrix.shape[0]) , dtype = float) ) 
    #K = np.subtract(matrix,np.dot(N,matrix))
    #K = np.subtract(K,np.dot(matrix,N))
    #K = np.add(K,np.dot(N, np.dot(matrix,N))) 
    
    # Stop the time
    timeBefore = time.time()
    # Compute n x n Gram Matrix K where K(i,j) = k(x_i,x_j)
    # Now we need to store our constants
    numberOfData = matrix.shape[0]
    # Now we need variables to hold the extracted components
    idx = 0
    idxVectors = np.zeros((featureNumber, dataset.transpose().shape[0]), np.double)
    # Now we can start to extract features
    for i in range(featureNumber):
        # Extract i-th feature using (11)
        maxValue = 0
        maxIn = 0
        for j in range(dataset.transpose().shape[0]):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if K[j][j] <= delta:
                continue
            sumOf = sum(np.multiply(K[j][:],K[j][:]))
            sumOf = (1/(dataset.transpose().shape[0]*K[j][j])) * sumOf
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        idxVectors[i] = K[maxIn] / np.linalg.norm(K[maxIn][:], 2)
        idx = maxIn
        # Now we must use equation (10) to update the Kernel Matrix K       
        K_new = np.zeros((dataset.transpose().shape[0],dataset.transpose().shape[0]), np.double)
        for j in range(dataset.transpose().shape[0]):
            for k in range(dataset.transpose().shape[0]):
                K_new[j][k] = K[j][k] - ((K[j][idx]*K[k][idx])/K[idx][idx])
        K = K_new
        
    timeAfter = time.time()
    #print("It took that many seconds to compute the data with AKFA:")
    #print(timeAfter-timeBefore)
    #finalData = Util.projectKernelComp(dataset, dataset, idxVectors, 'gauss')
    #return finalData, idxVectors
    return idxVectors


    