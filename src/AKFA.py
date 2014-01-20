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

# -------------------- Code -------------------------#

def akfa(dataset, featureNumber=2, delta = 0.0,kernel='gauss'):
    # Here we need to transform the matrix, so that we can access the vectors easier
    #atrix = dataset.transpose()
    # Stop the time
    timeBefore = time.time()
    # Compute n x n Gram Matrix K where K(i,j) = k(x_i,x_j)
    K = Util.comp_K(dataset.transpose(), kernel)
    # Now we need to store our constants
    numberOfData = dataset.transpose().shape[0]
    # Now we need variables to hold the extracted components
    idx = 0
    idxVectors = np.zeros((featureNumber, numberOfData), np.double)
    # Now we can start to extract features
    for i in range(featureNumber):
        # Extract i-th feature using (11)
        maxValue = 0
        maxIn = 0
        for j in range(numberOfData):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if K[j][j] <= delta:
                continue
            sumOf = sum(np.multiply(K[j][:],K[j][:]))
            sumOf = (1/(numberOfData*K[j][j])) * sumOf
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        idxVectors[i] = K[maxIn] / np.linalg.norm(K[maxIn][:], 2)
        idx = maxIn
        # Now we must use equation (10) to update the Kernel Matrix K       
        K_new = np.zeros((numberOfData,numberOfData), np.double)
        for j in range(numberOfData):
            for k in range(numberOfData):
                K_new[j][k] = K[j][k] - ((K[j][idx]*K[k][idx])/K[idx][idx])
        K = K_new
        
    timeAfter = time.time()
    print("It took that many seconds to compute the data with AKFA:")
    print(timeAfter-timeBefore)
    finalData = Util.projectKernelComp(dataset, dataset, idxVectors, kernel)
    return finalData, idxVectors


# ---------------- Testing ------------------#

if __name__ == '__main__':
    
    x = Util.gen_Circle();    
    #x = np.array( ( (1,2,3,4,5), (-1,3,-5,8,3) ), np.double)
    figure(0)
    title("Original Data")
    plot(x[0][range(30)], x[1][range(30)], 'ro')
    plot(x[0][range(30,90)], x[1][range(30,90)], 'ro',color='blue')
    plot(x[0][range(90,150)], x[1][range(90,150)], 'ro',color='green')

    finalData, comps = akfa(x,2)
    #print(comps)
    #np.savetxt("CircMatrix1.txt", x, fmt='%.18e', delimiter=',', newline='\n')
    #np.savetxt("CircMatrixResults1.txt", comps, fmt='%.18e', delimiter=',', newline='\n')
    figure(1)
    title("Data after projecting")
    plot(finalData[0][range(30)], finalData[1][range(30)], 'ro')
    plot(finalData[0][range(30,90)], finalData[1][range(30,90)], 'ro',color='blue')
    plot(finalData[0][range(90,150)], finalData[1][range(90,150)], 'ro',color='green')

    show()

    
    