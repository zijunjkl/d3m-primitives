#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learns the all discrete fully observed DAGs parameters. 

dataMatrix: NxD matrix last column is the class label
D:          number of R.V's

listRV:     list D elements each element is a NxK_i bool matrix
            for i'th item (r,c) of the matrix denotes whether X_n_i = c
            
parents:    list D elements each element is a N_pa_i, vector
            for i'th item the vector denotes the indexs of the parents
stateNo:    D, vector ith item denotes the number of states of X_i       

CPD:        list D elements each one stores a N_pa_i+1 dimensional matrix 
            for i'th element each matrix dims is NxK_pa_i_1x...xK_pa_i_N_pa_i
stateNo:    size D vector state[i] = K_i
stateDicList:list D elements. Each element dict. Dict[i] = s_i            
"""

import numpy as np
from rpi_d3m_primitives.structuredClassifier.helper import getlistRV, getNewState, setDimWithIndx, getWithIndxArray
import copy
#getlistRV, getNewState, changeCPD

def learnCPDAllD( dataMatrix, stateNo, parents, alpha, N0, debug = False, weighted = False, wVec = [], bayesInf = False, PointInf = False):
    
    N = np.size( dataMatrix,0)
    D = len( parents)
    #Initialize CPD arrays 
    CPD = []
    check = []
    
    for i in range(D):
        
        indx = np.insert( parents[i], 0, i).astype( int) # [indx of node, index of parents]
        CPD.append( np.zeros( stateNo[indx]))
#        check.append( np.zeros( stateNo[parents[i]]))
        
    listRV = getlistRV( dataMatrix, stateNo)
    
    for i in range( D):
        
        missingVal = 0
        indxI = listRV[i]
        parentIndx = parents[i]
        N_pa_i = np.size( parentIndx)
        subStateNo = np.zeros( N_pa_i)
        
        for j in range( N_pa_i):
    
            subStateNo[j] = stateNo[parentIndx[j]]
            
        noStates = np.prod( subStateNo).astype(int)
        
        
        if noStates != 1: #Todo check empty array instead one state feasbible?
            
            indxParState = np.zeros( [N, N_pa_i]).astype( bool)
            state = np.zeros( N_pa_i).astype( int)  
            
            for count in range( noStates):
            
                for j in range( N_pa_i):
                    
                    indxParState[:,j] = listRV[parentIndx[j]][:,state[j]]
                
                #N,
                stateConfirm = np.prod( indxParState,1).astype( bool)
                #NxK_i
                stateApply = np.logical_and( indxI, 
                                            np.tile( np.expand_dims(
                                                    stateConfirm,1),[1,stateNo[i]]))
                if np.sum( stateConfirm,0) != 0:
                    
                    if weighted == 0:
                        #K_i,
                        counts = np.sum( stateApply,0)
                        #K_i x1 olmali # Check type make sure 0.
                        if not bayesInf: 
                            # MLE
                            estimatedPi = np.expand_dims(counts / np.sum(counts),1)
                        else:
                                                    
                            Nt = np.sum( counts)
                            if PointInf:
                                # this part is for MAP point inference
                                estMLE = counts/Nt
                                estPrior = 0
                                estPrior = np.sum(indxI,0) / np.size(indxI,0)       #K_i                                     
                                post = Nt/(Nt+N0)*estMLE + N0/(Nt+N0)*estPrior         #K_i                             
                            elif not PointInf:
                                #this line is for bayesian inference
                                post = (counts + alpha)/(Nt + alpha * stateNo[i])#bayesian inference
                            estimatedPi = np.expand_dims(post,1)   
                    else:
                        
                        counts = np.zeros( np.size(stateApply,1))
                        for k in range( np.size(stateApply,1)):
                                
                            counts[k] = np.sum( wVec[ stateApply[:,k]])
                        estimatedPi = np.expand_dims(counts / np.sum(counts),1)
                             
    #                changeCPD( CPD, i, state, estimatedPi)
                    setDimWithIndx( CPD[i], state, 0, estimatedPi)
                    
                else:
                    if debug:
                        missingVal +=1
#                        print('Putting uniform since no state')
                    #Put uniform and put a check on 
                    if not bayesInf:
                        # MLE
                        estimatedPi = np.ones([stateNo[i],1])/stateNo[i]
                    else:
                        
                        if PointInf:
                            # this part is for MAP point inference
                            estPrior = np.sum(indxI,0) / np.size(indxI,0)       #K_i                                     
                            post = estPrior                                     #K_i                            
                            estimatedPi = np.expand_dims(post,1)
                        elif not PointInf:
                            #this line is the new one for bayesian inference. 
                            estimatedPi = np.ones([stateNo[i],1])/stateNo[i]#Bayesian inference same as MAP                            
                        
                    setDimWithIndx( CPD[i], state, 0, estimatedPi)
                    #Write setIndx
#                    setIndx( check[i], state, 0, 1)

                state = getNewState( state, subStateNo)

        else:
            
            counts = np.sum( indxI,0)
            estimatedPi = np.expand_dims(counts / np.sum(counts),1)
#            changeCPD( CPD, i, state, estimatedPi)
            CPD[i] = estimatedPi[:,0]
#            setDimWithIndx( CPD[i], state, 0, estimatedPi)
        if debug:   
             check.append(str(missingVal) + '/' + str(noStates))
             #print( str(missingVal) + '/' + str(noStates))
        
    if debug == False:
        return CPD
    else:
        return CPD, check
    
def getLogLikelihood( dataMatrix, parents, CPD):
    
    #Calculates the log likelihood of the dataMatrix
    
    D = np.size( dataMatrix, 1)
    evidence = dataMatrix
    
    logcondP = 0.
    for i in range(D):
        
        obsNodeList = copy.deepcopy(parents[i])     #Getting the parent nodes, global indx
        obsNodeList.insert(0,i)                     #Adding the node itself, global indx
        obsNodeEvi = evidence[:,obsNodeList]
        #+1 is because infNodeIndx[i] returns the index of the inferred node among the parents even though it's 0'th paretnt iin a CPD 0'th dim is node itself
        temp = getWithIndxArray( CPD[i], obsNodeEvi)
        logcondP += np.log(temp)
    
    return np.sum( logcondP)

def getTotalPar( CPD):
    
    #Find the total number of parameters
    
    totalPar = 0
    D = len( CPD)
    for i in range( D):
        
        shape = np.asarray(np.shape(CPD[i]))
        shape[0] -= 1                       # -1 since sums to one
        totalPar += np.prod( shape)
        
    return totalPar

def expandCPD( i, A, D, stateNo, parents):
    
    #Expand the CPD of i'th node to the same dimension of joint distribution
    #Kevin Murphy calls this Factor Table or smh
    
    shape = np.asarray(A.shape)
    ndim = np.size(shape)
    A = A.reshape( np.concatenate(
            (shape, np.ones(D - ndim, dtype=int))))
    
    #global Indx of the CPD variables P(X_i \mid X_pa_i) i, pa_i_1, ... , pa_i_N_i
    globalIndx = copy.deepcopy(parents[i]) 
    globalIndx.insert(0, i)      
    currPos = np.arange( len(globalIndx))
    
    for i in range( len(globalIndx)):
        
        ordered = np.arange( 0, D, dtype = int)
        ordered[currPos[i]] = globalIndx[i]
        ordered[globalIndx[i]] = currPos[i]
        if globalIndx[i] <= len(globalIndx)-1: #Then we are switching with current CPDs
            currPos[globalIndx[i]] = currPos[i]        
        A = np.transpose(A,ordered)
    
    selOther = np.ones( [D], bool)
    selOther[globalIndx] = 0    
    ordered = np.arange( 0, D, dtype = int)
    other = ordered[selOther]
    
    for indx in other:
        
        A = np.repeat( A, stateNo[indx], indx)
    
    return A


def getJointD( CPD, stateNo, parents):
    
    #Get's the joint distribution
    
    jointCPD = np.ones(stateNo, dtype = np.float16)
    D = len( CPD)
    
    for i in range( D):
        
        expCPD = expandCPD(i, CPD[i], D, stateNo, parents)
        jointCPD = np.multiply( jointCPD, expCPD)
    
    return jointCPD

def initCPDAllD( parents, stateNo, uniform = True):
    
    D = len( parents)
    #Initialize CPD arrays 
    CPD = []
    
    for i in range(D):
        
        if uniform:
            
            indx = np.insert( parents[i], 0, i).astype( int) # [indx of node, index of parents]
            CPD.append( np.ones( stateNo[indx])/stateNo[i])
            
        else:
#        CPD.append( np.zeros( stateNo[indx]))
            CPD.append( np.random.dirichlet(
                    np.ones(stateNo[i]), 
                    stateNo[parents[i]]).T)
            
    return CPD

def printCPD( CPD, letters, parents, stateNo):
    
    D = len(CPD)
    
    for i in range( D):
    
        print('Node ' + letters[i])
        
        if len( CPD[i].shape) == 1: # Has 0 parents
            
            print(CPD[i])
            
    
        if len( CPD[i].shape) == 2: # Has one parent
            
            parent = parents[i][0]
            
            for j in range(stateNo[ parent]):
                
                print('Parent Node ' + letters[parent] + '=' + str(j))
                print(CPD[i][:,j])
    
        if len( CPD[i].shape) == 3: # Has two parents 
                
            for j in range( stateNo[parents[i][0]] ):
                
                print('Parent Node ' + letters[parents[i][0]] + '=' + str(j))
                
                for k in range( stateNo[parents[i][1]] ):
    
                    print('Parent Node ' + letters[parents[i][1]] + '=' + str(k))
                    print(CPD[i][:,j,k])
    
        
        
