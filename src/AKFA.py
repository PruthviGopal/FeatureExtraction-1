#!/usr/bin/python
'''

This file supports a module for doing ACCELERATED KERNEL FEATURE ANALYSIS on a given matrix

@author: Christoph Rauterberg
'''



# ---------- Imports ----------#
import numpy as np
from pylab import *
import time
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances as padis
import copy
# -----------------------------#



# -------------------- Code  -------------------------#
def buildGramMatrix(dataset, n_samples,chunkSize, sigma):
    """ Compute the Gram Matrix of a given data set using a gaussian kernel.

        Parameters
        ----------
        dataset: array-like, shape (n_samples, n_dimensions)
            Data set, where n_samples in the number of samples
            and n_dimensions is the number of dimensions.
            
        n_samples: number of data points in the data set
        
        chunkSize: The block size for building the Gram Matrix.
            Restrain: n_samples % chunkSize = 0 !!
            
        sigma: Value used by the Gaussian Kernel
            Default: 4

        Returns
        -------
        kernelMatrix : numpy.array, shape (n_samples, n_samples)
            The computed Gram Matrix
    """
    if n_samples % chunkSize != 0:
        raise ValueError('The chunkSize must be a true divider of the number of samples')
    kernelMatrix = np.ones( (n_samples,n_samples), dtype=np.float16)
    print(".....")
    print("Creating new Sparse Matrix for holding the Gram Matrix")
    print("For a large data set, this may take a while ...")
    chunk = n_samples/chunkSize
    print("Chunking data set into %d different sets" % (chunkSize))
    print(".... now starting to compute %d x %d blocks of Gram Matrix" % (chunk,chunk) )
    timeBef = time.time()
    for i in range(chunkSize):
        for j in range(chunkSize):
            kernelMatrix[(i*chunk):((i+1)*chunk),(j*chunk):((j+1)*chunk)] = exp( - padis(dataset[(i*chunk):((i+1)*chunk),:], dataset[(j*chunk):((j+1)*chunk),:],metric='euclidean')**2 / (2*sigma*sigma))
        print("  ... Finished the first %d rows" % ((i+1)*chunk))
    print("It took %f to compute the Gram matrix" % (time.time()-timeBef))
    print(".....")
    print("Therefore: Done computing the Gram Matrix")
    print("WUHU!!")
    return kernelMatrix

def projectKernelComp(dataset,comp,sigma=4):
    """ Project the given data sets onto the given components.

        Parameters
        ----------
        dataset: array-like, shape (n_samples, n_dimensions)
            Data set, where n_samples in the number of samples
            and n_dimensions is the number of dimensions.
        
        comp: array-like, shape (n_features,n_samples)
            Matrix that holds the extracted vectors
            
        sigma: Value used by the Gaussian Kernel
            Default: 4

        Returns
        -------
        fData : csr_matrix
            A sparse matrix containing the projected data set of shape (n_samples, n_features)
    """
    numberOfDataPoints = dataset.shape[0]
    numberOfFeatures = comp.shape[0]
    print(".. read Matrix contains %f features" %(numberOfFeatures))
    fData = sparse.lil_matrix( (numberOfDataPoints,numberOfFeatures),dtype = np.float16)
    print(" ... beginning to project components onto data set")
    #now, we must project the chosen components onto the dataset
    for x in range(numberOfDataPoints):
        #We have now chosen a point in our dataset which is to be adapted
        timePro = time.time()
        for index in range(numberOfFeatures):
            fData[x,index] = np.dot(comp[index],exp( - padis(dataset, dataset[x,:],metric='euclidean')**2 / (2*sigma*sigma)))
        #print("Finished for point %d in %f" %(x,time.time() - timePro))

    print("....... projecting finished!")
    return fData.tocsr()


def akfa(dataset, n_features=2, delta = 0.0, sigma=4, chunkSize = 5,isMatrixGiven=False, K=None):
    """ Computes the Principal Components of a given data set.

        Parameters
        ----------
        dataset: array-like, shape (n_samples, n_dimensions)
            Data set, where n_samples in the number of samples
            and n_dimensions is the number of dimensions.
        
        n_features: The number of features that are to 
            be extracted from the data setz
            Default: 2
            
        delta: Value used for the Cut-Off Version of AKFA
            Default: 0.0 (for non-cut-off-Version)
            
        sigma: Value used by the Gaussian Kernel
            Default: 4
            
        chunkSize: Size of the blocks of the Gram Matrix that
            are to be calculated
            Default: 5 (for Standart-Testing-Set with 600 samples points
        
        isMatrixGiven: Boolean indicating if the Gram Matrix has already
            been build.
            Default: False
            
        K: Pre-calculated Gram Matrix for the given data set
            Only used if isMatrixGiven is True

        Returns
        -------
        finalData : csr_matrix, shape ( n_samples, n_features )
            A sparse matrix containing the projected data set of shape (n_samples, n_features)
            
        comps: numpy-array, shape ( n_features, n_samples )
            A Matrix holding the extracted components
            
        References
        ----------
        Accelerated Kernel Feature Analysis was intoduced in:
            Xianhua Jiang, Robert R. Snapp, Yuichi Motai, 
            and XingquanZhu. 2006. Accelerated Kernel 
            Feature Analysis. In Proceedings of the 2006 IEEE 
            Computer Society Conference on Computer IEEE 
            Vision and Pattern Recognition, IEEE.
    """
    allTime = time.time()
    print("___________")
    print("Setting up for Feature Extraction via Accelerated Kernel Feature Analysis")
    print(" ... %d features are to be extracted!" %(n_features))
    print("Data set begin loaded...")
    print(".....")
    print(".....")
    print("Loading of data set is completed")
    print(".....")
    n_samples = dataset.shape[0]
    n_dimensions = dataset.shape[1]
    if isMatrixGiven:
        kernelMatrix = K
    else:
        kernelMatrix = buildGramMatrix(dataset,n_samples,chunkSize, sigma)
    # Now we can compute the Gram Matrix
    print("The loaded file contains %d samples points and %d dimensions " % (n_samples,n_dimensions ))
    print(".....")
    
    print("......")
    print("............")
    print("Will continue with extracting features now!")
    # Stop the time
    timeBefore = time.time()
    # Now we need variables to hold the extracted components
    idx = 0
    idx_vectors = np.zeros( (n_features,n_samples),dtype = np.float16)

    # Now we can start to extract features
    for i in range(n_features):
        print("___________________")
        print("Starting extracting of feature %d" % (i+1))
        # Extract i-th feature using (11)x
        maxValue = 0
        maxIn = 0
        timeIn = time.time()
        print("....")
        for j in range(n_samples):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if kernelMatrix[j,j] <= delta:
                continue
            #sumOf = (1/(n_samples*K[j,j])) * K.getcol(j).multiply(K.getcol(j)).sum()
            sumOf = (1/(n_samples*kernelMatrix[j,j])) * np.multiply(kernelMatrix[j,:],kernelMatrix[j,:]).sum()
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        # NEED TO CHECK FOR ZERO VALUES IN DATA
        print("........")
        idx_vectors[i,:] =  kernelMatrix[maxIn,:] / np.linalg.norm(kernelMatrix[maxIn,:])
        idx = maxIn
        print("__Feature found!")
        print("_____ ... which took %f " % (time.time()-timeIn))
        if i == n_features-1:
            continue
        idxVec = copy.copy(kernelMatrix[:,idx])

        print("Now updating the Gram Matrix")
        timeBef = time.time()
        # Now we must use equation (10) to update the Kernel Matrix K
        for i in range(n_samples):
            fak = idxVec[i]/idxVec[idx]
            kernelMatrix[:,i] = np.subtract(kernelMatrix[:,i],idxVec*fak)
        #K_new = sparse.dok_matrix( (n_samples,n_samples),dtype = np.double)
        #print("Updating Gram Matrix!")
        #for j in range(n_samples):
            #for k in range(n_samples):
                #K_new[j,k] = K[j,k] - ((K[j,idx]*K[k,idx])/K[idx,idx])
        #K = K_new.tocsr()
        #K.eliminate_zeros()
        print("Update successfull - in %f " % (time.time()-timeBef))
        print(" ------> continue!")
    

    print("__________")
    print("It took that many seconds to compute the data with AKFA: %f" % (time.time()-allTime))
    print("__________")
    #finalData = projectKernelComp(dataset, idx_vectors)
    #print("__________")
    #print("It took that many seconds to project the components onto the data: %f" % (time.time()-tmpTime))
    #print("__________")
    return idx_vectors


    
