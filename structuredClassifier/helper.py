# -*- coding: utf-8 -*-

import numpy as np
#from scipy.io import loadmat

def getdataMatrix( ):
    
    s = loadmat('discrete_2000.mat')
    dataMatrix = s['discrete_data']
    dataMatrix = dataMatrix.astype(int)-1 # In python I start from 0
    
    return dataMatrix

def getdataMatrix2( ):
    
    s = loadmat('gaussian_2000.mat')
    dataMatrix = s['gaussian_data']    
    return dataMatrix

def getlistRV( dataMatrix, stateNo):
    
    N = np.size( dataMatrix, 0)
    D = np.size( dataMatrix, 1)
    listRV = []
    for i in range( D):
        
        indxMatrix = np.zeros( [N, stateNo[i]]).astype(bool)
        tempVec = dataMatrix[:,i]
        
        for j in range( stateNo[i]):
            
            boolVec = tempVec == j
            indxMatrix[:,j] = boolVec
        
        listRV.append( indxMatrix)

    return listRV

def getNewState( oldState, subStateVec):
    
    #Gets the old state and adds one 
    newState = np.copy(oldState)    
    
    curAxis = 0
    stop = False
    
    #Check for overheads
    while curAxis != np.size( subStateVec) and not stop :

        newState[curAxis] +=1

        if newState[curAxis] == subStateVec[curAxis]: #Time to set 0 and incerement currAxis
            
            newState[curAxis] = 0
            curAxis += 1
            
        else:
            stop = True
            
    return newState
    
def changeCPD( CPD, i, state, estimatedPi):
    
    #To do : Write this code generalizable for any number of parents

    setDimWithIndx( CPD[i], state, 0, estimatedPi)
    
def getIndx( A, indx):
    
#    returns A[ indx[0], indx[1], ... , indx[N-1]]
    
    shapeA = np.flip( np.asarray(np.shape(A)))
    magnitude = np.zeros( np.size(indx)).astype(int)
    #    magnitude = np.array([12,4,1])
    for i in range( np.size(indx)):
        
        magnitude[i] = np.prod(shapeA[:-(i+1)])
        if i == np.size(indx)-1:
            magnitude[i] = 1
    B = A.flat
    indx2 = np.sum(indx*magnitude)
    return B[indx2]

def getDimWithIndx( A, indx, d):

#   Assuming len(indx) = A.ndim - 1
#   Assuming indx[j] corresponds to jth dim 
#   Gets the all items in the unspecied dimension
#   where indx of all other dimensions are specified.
#   A[indx[0], indx[1], ..., indx[d-1], :, indx[d+1], ...]

    shapeAflip = np.flip( np.asarray(np.shape(A)))
    magnitude = np.zeros( A.ndim).astype(int)
    for i in range( A.ndim):
        
        magnitude[i] = np.prod( shapeAflip[:-(i+1)])
        if i == A.ndim-1:
            magnitude[i] = 1

    B = A.flat
    dMag = magnitude[d]
    
    tempArr = np.arange( 0,np.size(magnitude))
    remMag = magnitude[tempArr[tempArr != d]] #Magnitude of the all indexes expect d
    constantIndx = np.sum( indx*remMag)
    indx2 = np.arange( 0,np.shape(A)[d])*dMag + constantIndx 
    
    return B[indx2]

#def setIndx( A, indx, d, newValues):
#    
#    #Write
        
def setDimWithIndx( A, indx, d, newValues):

#   Assuming len(indx) = A.ndim - 1
#   Assuming indx[j] corresponds to jth dim 
#   Gets the all items in the unspecied dimension
#   where indx of all other dimensions are specified.
#   A[indx[0], indx[1], ..., indx[d-1], :, indx[d+1], ...]
    
    if len(indx) == A.ndim -1:
        shapeAflip = np.flip( np.asarray(np.shape(A)))
        magnitude = np.zeros( A.ndim).astype(int)
        for i in range( A.ndim):
            
            magnitude[i] = np.prod( shapeAflip[:-(i+1)])
            if i == A.ndim-1:
                magnitude[i] = 1
    
        B = A.flat
        dMag = magnitude[d]
        
        tempArr = np.arange( 0,np.size(magnitude))
        remMag = magnitude[tempArr[tempArr != d]] #Magnitude of the all indexes expect d
        constantIndx = np.sum( indx*remMag)
        indx2 = np.arange( 0,np.shape(A)[d])*dMag + constantIndx 
        
        B[indx2] = newValues
    else:
        print('ERROR CONDITION NOT SATISFIED')
    
def getDimWithIndxArray( A, indx, d):

#   indx: N_repeats x A.ndim - 1
#   Assuming len(indx) = A.ndim - 1
#   Assuming indx[j] corresponds to jth dim 
#   Gets the all items in the unspecied dimension
#   where indx of all other dimensions are specified.
#   A[indx[0], indx[1], ..., indx[d-1], :, indx[d+1], ...]

    shapeAflip = np.flip( np.asarray(np.shape(A)))
    magnitude = np.zeros( A.ndim).astype(int)           # ndim, 
    
    for i in range( A.ndim):
        
        magnitude[i] = np.prod( shapeAflip[:-(i+1)])
        if i == A.ndim-1:
            magnitude[i] = 1

    B = A.flat
    dMag = magnitude[d]
    
    tempArr = np.arange( 0, np.size(magnitude))         # ndim,
    remMag = magnitude[tempArr[tempArr != d]]           # ndim-1,
                                                        #Magnitude of the all indexes expect d
                                                                                                               
    constantIndx = np.matmul( indx, remMag)             # N_repeats,
    indxD = np.arange( 0, np.shape(A)[d])*dMag          # shape[d],
    indxMatrix = np.repeat(np.expand_dims(indxD,0), 
                           np.size(indx,0), 0)          # N_repeats, shape[d]
    indxMatrix = indxMatrix + np.repeat(np.expand_dims(constantIndx,1),
                                        np.shape(A)[d],1) #N_repeats, shape[d]
    indxMatrix = indxMatrix.astype(int)
    C = B[indxMatrix]

    return C

def getWithIndxArray( A, indx):

#   indx: N_repeats x A.ndim
#   Assuming len(indx) = A.ndim 
#   Assuming indx[j] corresponds to jth dim 
#   A[indx[0], indx[1], ..., indx[d-1], indx[d], indx[d+1], ...]

    shapeAflip = np.flip( np.asarray(np.shape(A)))
    magnitude = np.zeros( A.ndim).astype(int)           # ndim, 
    
    for i in range( A.ndim):
        
        magnitude[i] = np.prod( shapeAflip[:-(i+1)])
        if i == A.ndim-1:
            magnitude[i] = 1
    
    B = A.flat
    constantIndx = np.matmul( indx, magnitude)             # N_repeats,
    C = B[constantIndx]

    return C

def totuple(a):
    #From stackedexchange
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
