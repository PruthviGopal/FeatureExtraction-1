'''
Created on 10.01.2013

This module contains all support methods used in the ComponentAnalysis

@author: Christoph Rauterberg
'''

# --------------- Imports ---------------
import numpy as np
import copy
import math
import random
from collections import deque

# ----------------- Code -----------------

'''
This function computes the mean over each vector of a given matrix
'''
def mean_mat(matrix):
    # To start with, compute a copy of the matrix that we can change
    norm_mat = copy.copy(matrix)
    # now we need a matrix containing ones, to later stratch the mean vec
    one = np.ones(norm_mat.shape, float)
    # compute the mean of each row, get a vector
    mean = norm_mat.mean(axis=1)
    # continue with computing a matrix holding the mean of each row, as the value of each row
    for i in range(matrix.shape[0]):
        norm_mat[i] = np.multiply(mean[i],one[i])
    # all thats left: substract the means from the given matrix
    norm_mat = np.subtract(matrix,norm_mat) 
    return norm_mat

'''
 This function reads a matrix written in the file Matrix.txt
 '''
def read_Matrix():
    mat = []
    with open('Matrix.txt', 'r') as file:
        for line in file:
            line = line.strip()
        if len(line) > 0:
            mat.append(map(float, line.split(',')))
    matrix = np.array(mat, dtype=float)
    return matrix;

'''
This function is used to center a matrix in feature space
'''
def center_K(matrix):
    # do this step by step, otherwise you will encounter problems due to bracets!
    N = comp_N(matrix.shape[0])
    K = np.subtract(matrix,np.dot(N,matrix))
    K = np.subtract(K,np.dot(matrix,N))
    K = np.add(K,np.dot(N, np.dot(matrix,N)))   
    return K

'''
This function is used to compute the matrix N used in the centering process
'''
def comp_N(size):
    return np.multiply(1./size,np.ones( (size,size) , dtype = float) )


# --------------- Kernel-Functions ---------------


def polynomial(vec_one, vec_two,d):
    return math.pow(np.dot(vec_one,vec_two), d)
    
def linear(vec_one,vec_two):
    return polynomial(vec_one,vec_two,1)


def exponential(vec_one, vec_two):
    return math.pow((np.dot(vec_one,vec_two) +1), 2)

#This function describes the Gaussian Kernel
def gaussian(vec_one,vec_two, sigma=4):
    return math.exp(-np.linalg.norm(vec_one - vec_two,2)**2/(2*sigma*sigma))

# ----------------------------------

def gen_Circle(small = 30, med = 60, big = 60, delta = 1, smallMax = 1, medMax = 4, bigMax = 8):
    val = np.zeros( (small+med+big,2),dtype=np.double);
    
    #y_val = np.array(range(small+med+big),dtype=np.double);
    
    random.seed()
    i = 0;
    while i < small:
        x = random.uniform(-smallMax,smallMax)
        y = random.uniform(-smallMax,smallMax)
        if x**2+y**2 <= smallMax*smallMax:
            val[i,0] = x
            val[i,1] = y
            i = i + 1
                    
    
    while i < (small+med):
        x = random.uniform(-medMax,medMax)
        y = random.uniform(-medMax,medMax)
        if (x**2+y**2 >= (medMax-delta)*(medMax-1)) & (x**2+y**2 <= medMax*medMax):
            val[i,0] = x
            val[i,1] = y
            i = i + 1
       
    while i < (small+med+big):
        x = random.uniform(-bigMax,bigMax)
        y = random.uniform(-bigMax,bigMax)
        if (x**2+y**2 >= (bigMax-delta)*(bigMax-1)) & (x**2+y**2 <= bigMax*bigMax):
            val[i,0] = x
            val[i,1] = y
            i = i + 1
    
    return val
        


