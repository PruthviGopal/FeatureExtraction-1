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
from scipy.sparse.linalg import eigs as eigenvec
from sklearn.metrics.pairwise import pairwise_distances as padis
import copy
# -----------------------------#
def oldKernelMatrix(dataset, n_samples,chunkSize, sigma):
    kernelMatrix = np.ones( (n_samples,n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_samples):
            kernelMatrix[i,j] =  exp( - (np.linalg.norm(np.subtract(dataset[i,:], dataset[j,:])))**2 / (2*sigma*sigma))
    
    return kernelMatrix




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
    kernelMatrix = np.ones( (n_samples,n_samples), dtype=np.float32)
    chunk = n_samples/chunkSize
    for i in range(chunkSize):
        for j in range(chunkSize):
            kernelMatrix[(i*chunk):((i+1)*chunk),(j*chunk):((j+1)*chunk)] = exp( - padis(dataset[(i*chunk):((i+1)*chunk),:], dataset[(j*chunk):((j+1)*chunk),:],metric='euclidean')**2 / (2*sigma*sigma))
        print("  ... Finished the first %d rows" % ((i+1)*chunk))
    return kernelMatrix

def projectKernelComp(dataset,comp,sigma=4):
    """ Project the given data sets onto thK_COMP_SUCCESSe given components.

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
    fData = sparse.lil_matrix( (numberOfDataPoints,numberOfFeatures),dtype = np.float16)
    for x in range(numberOfDataPoints):
        for index in range(numberOfFeatures):
            fData[x,index] = np.dot(comp[index],exp( - padis(dataset, dataset[x,:],metric='euclidean')**2 / (2*sigma*sigma)))
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
            
        idx_vectors: numpy-array, shape ( n_features, n_samples )
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
    n_samples = dataset.shape[0]
    n_dimensions = dataset.shape[1]
    if n_dimensions < n_features:
        raise ValueError("Can only extract n_dimensions features at maximum - trying to extract too many features")
    if isMatrixGiven == True:
        kernelMatrix = K
    else:
        kernelMatrix = buildGramMatrix(dataset,n_samples,chunkSize, sigma)
    idx = 0
    idx_vectors = np.zeros( (n_features,n_samples),dtype = np.float16)

    for i in range(n_features):
        maxValue = 0
        maxIn = 0
        for j in range(n_samples):
            sumOf = 0
            # Neglect all cases, where diagonal element is smaller then delta
            # delta set to 0.0 therefore we need to set an smaller and equal then
            if kernelMatrix[j,j] <= delta:
                continue
            sumOf = (1/(n_samples*kernelMatrix[j,j])) * np.multiply(kernelMatrix[j,:],kernelMatrix[j,:]).sum()
            if ( sumOf > maxValue):
                maxValue = sumOf
                maxIn = j
        idx_vectors[i,:] =  kernelMatrix[maxIn,:] / np.linalg.norm(kernelMatrix[maxIn,:])
        idx = maxIn
        if i == n_features-1:
            continue
        idxVec = copy.copy(kernelMatrix[:,idx])
        tmpK = np.ones( (n_samples,n_samples), dtype=np.float32)
        for k in range(n_samples):
            #for j in range(n_samples):
                #tmpK[k,j] = K[k,j] - (K[j,idx]*K[k,idx])/K[idx,idx]
            fak = idxVec[k]/idxVec[idx]
            kernelMatrix[:,k] = np.subtract(kernelMatrix[:,k],idxVec*fak)
        #K = tmpK
    return idx_vectors

def kpca(dataset, n_features=2, sigma=4, chunkSize = 5,isMatrixGiven=False, K=None):
    """ Computes the Principal Components of a given data set using Kernel Principal Component Analysis.

        Parameters
        ----------
        dataset: array-like, shape (n_samples, n_dimensions)
            Data set, where n_samples in the number of samples
            and n_dimensions is the number of dimensions.
        
        n_features: The number of features that are to 
            be extracted from the data setz
            Default: 25
            
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
        val : ndarray, shape ( n_samples )
            An array holding the corresping eigenvalues
            
        eig: ndarray, shape ( n_features, n_samples )
            A Matrix holding the extracted components
            
        References
        ----------
        Kernel Principal Component Analysis was intoduced in:
            
    """
    n_samples = dataset.shape[0]
    n_dimensions = dataset.shape[1]
    if n_dimensions < n_features:
        raise ValueError("Can only extract n_dimensions features at maximum - trying to extract too many features")
    if isMatrixGiven:
        kernelMatrix = K
    else:
        kernelMatrix = buildGramMatrix(dataset,n_samples,chunkSize, sigma)
        
    val, eig = eigenvec(kernelMatrix, n_features)
    return  eig, val


    
