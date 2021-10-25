#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input:          
        testMatrix:     NxD-1 matrix
        parents:            D list // each element is a list
        CPD:                D list // each element is matrix with dim = N_pa_i+1 size(, i) = K_g_i           
        inferNode:          integer // selected Node to get post. prob.
"""

import copy 
import numpy as np
from rpi_d3m_primitives.structuredClassifier.helper import getDimWithIndxArray, totuple, getIndx
from rpi_d3m_primitives.structuredClassifier.helperData import getChild#getlistRV, getNewState, changeCPD

def posteriorInf( testMatrix, parents, CPD, inferNode, debug = False, normalize = False):
    
    #Do posterior inference P(X_inf \ mid X_/inf = evidence)
    #It's note normalized.
    
    import time
    start = time.time()
    
    #inferNode = 9
    childList, infNodeIndx = getChild( inferNode, parents)
    evidence = testMatrix
    
    parentList = parents[inferNode] 
    parentEvidence = evidence[:,parentList]
    
    condP = getDimWithIndxArray( CPD[inferNode], parentEvidence, 0)
    
    for i in range( len(childList)):
        
        obsNodeList = copy.deepcopy(parents[childList[i]])     #Getting the parent nodes
        del obsNodeList[infNodeIndx[i]]         #Removing the inferred node
        obsNodeList.insert(0,childList[i])      #The node itself is observed
        
        obsNodeEvi = evidence[:,obsNodeList]
        #+1 is because infNodeIndx[i] returns the index of the inferred node among the parents even though it's 0'th paretnt iin a CPD 0'th dim is node itself
        temp = getDimWithIndxArray( CPD[childList[i]], obsNodeEvi, infNodeIndx[i]+1)
        condP = condP * temp
        if debug == True: 
            print(np.sum(np.isnan(temp)))

        
    if debug == True: 
        backP = copy.deepcopy( condP)
        
    Yest = np.argmax(condP,1)
#    Yest[np.sum( condP,1) == 0] = -1
    
    if normalize:
        condP = condP / np.repeat( np.expand_dims(np.sum(condP,1),1), np.size(condP,1),1)
    
#    print (time.time() - start)
    

    if debug == True:    
        return condP, Yest, backP
    
    else:
        return condP, Yest
    
def enum( jointCPD, testMatrix, inferNode):
    
    #P( X_{-1} \mid  X_\{-1} = evide)
    #Enumeration when all R.V's expect the inferNode is observed
    
    condP = getDimWithIndxArray( jointCPD, testMatrix, inferNode)
    condP = condP / np.repeat( np.expand_dims(np.sum(condP,1),1), np.size(condP,1),1)
    
    Yest = np.argmax( condP, 1) 
    
    return condP, Yest

def enumUn( jointCPD, testMatrix2):
    
    #P(X_{-1} \mid X_s =evide ) where X_s is a subset of nodes X_\{-1}
    #Enumeration when there is a subset of variables unobserved
    
    unObs = np.arange(5,10) #Global indx of unobserved variables
    marginalCPD = np.sum( jointCPD, axis=totuple(unObs))
    condP = getDimWithIndxArray( marginalCPD, testMatrix2, marginalCPD.ndim-1)
    condP = condP / np.repeat( np.expand_dims(np.sum(condP,1),1), np.size(condP,1),1)

    Yest = np.argmax( condP, 1) 
    
    return condP, Yest

def posteriorInf2( jointState, jointD, infBool, mode = 1):
    
    #Multivariate 
    #Input:
    #       jointState : Dx1 assignment values
    #       infBool : Dx1 bool True if infBool[i] is inferred
    
    #Calculates P( X_i_1 = x_i_1, ..., X_i_N = x_i_N \mid X_o_1 = x_o_1, ..., X_o_N_o}  = x_o_N_o)
    #Assuming X_i, X_o = X_all 
    #Edit: jointState inferred variables values are dummy 
    
    D = jointD.ndim
    unObs = np.arange( 0,D, dtype = int)[infBool] #Global indx of unobserved variables
    marginalD = np.sum( jointD, axis=totuple(unObs), keepdims = True)

    tempState = copy.deepcopy(jointState)
    tempState[infBool] = np.zeros( np.sum(infBool))
#    tempState[infVars] = np.zeros( len(infVars))

    normal = getIndx( marginalD, tempState)
    
    posterior =  getIndx( jointD, jointState) / normal
    
    if mode == 1:
        return posterior
    
    else:
        return normal
