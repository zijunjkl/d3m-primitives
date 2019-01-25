#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:25:36 2018

@author: EceKoyuncu
"""

import pandas
import numpy as np
import subprocess

def getDataMatrix():
    
    df = pandas.read_csv('data.csv')
    
    dataMatrix = df.values
    dataMatrix = dataMatrix[:,1:]
    
    return dataMatrix


def getParent( D, graphName):
    
    df = pandas.read_csv( graphName + '.csv')
    parent = pandas.Series.tolist(df[df.columns.values[1]])
    child = pandas.Series.tolist(df[df.columns.values[2]])
#    parent = pandas.Series.tolist(df['from'])
#    child = pandas.Series.tolist(df['to'])
    
    #adjMatrix = np.zeros(D,D)
    parentList = []
    
    #Make it more efficienct
    for i in range(D):
        parentTemp = []     #Parents of i'th node
        for ii in range(len(child)):
            
            if child[ii] == i:
                parentTemp.append(parent[ii])
        
        parentList.append(parentTemp)

    return parentList


def getStateUpdateData( dataMatrix ):
    
    '''Each feature column may have an arbitrary states such as 2 and 4 of
    last column of breast instead of 0,1. 
    Returns:
        stateNo: size D vector state[i] = K_i
        stateDicList: list D elements. Each element dict. Dict[i] = s_i
    '''
    
    D = np.size( dataMatrix, 1)
    
    stateNo = np.zeros(D).astype(int)
    stateDicList = []               # list DxK_i each each item is a dictionary keys: 0,1,K_i   values: corresponding states
    
    
    for i in range( D):
            
        temp_unique = np.unique( dataMatrix[:,i])
        temp_ordered = np.arange( 0, np.size(temp_unique))
        stateDic = dict( zip( temp_ordered, temp_unique))
        stateDicList.append( stateDic)
        
        #Replacing the states with ordered values in dataMatrix
        for ii in range( np.size( temp_unique)):
            
            indx = dataMatrix[:,i] == temp_unique[ii]
            dataMatrix[indx,i] = temp_ordered[ii]
            
        stateNo[i] = np.size(temp_unique) # Storing K_i 

    return stateNo, stateDicList

def printCPD( D, CPD, parents, stateNo):
    
    letters = []
    for i in range(D):
        letters.append(str(i))
    
    for i in range(D):
        
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

                    
def getChild( nodeSelect, parents):
    
    # get the child nodes of a node
    D = len( parents)
    childList = []
    whichParentList = []
    for i in range( D):
        
        if i != nodeSelect:
            
            for ii in range( len(parents[i])):
                
                if nodeSelect == parents[i][ii]:
                
                    childList.append(i)
                    whichParentList.append(ii)
                    ii = len(parents[i]) # break
    
    return childList, whichParentList

def callRscript( slType):
    
    # Define command and arguments
    command = 'Rscript'
    path2script = 'deka.R'
    
    # Variable number of args in a list
    args = [slType]
    
    # Build subprocess command
    cmd = [command, path2script] + args
    
    # check_output will run the command and store to result
    x = subprocess.check_output(cmd, universal_newlines=True)
    
    return x 

def checkAtLeast2( trainMatrix):
    
    #Checks whether at least two state is observed for each R.V in each data set
    
    for i in range( np.size( trainMatrix,1)):
        
        uniqArr = np.unique( trainMatrix[:,i])
        if np.size( uniqArr ) == 1: #Only one state is observed in the training data not helpful
            
            return False
        
    return True
                
def getParentEdgeMatrix( D, edgeMatrix):
    
    parent = edgeMatrix[:,0]
    child = edgeMatrix[:,1]
    
    #adjMatrix = np.zeros(D,D)
    parentList = []
    
    #Make it more efficienct
    for i in range(D):
        parentTemp = []     #Parents of i'th node
        for ii in range(child.shape[0]):
            
            if child[ii] == i:
                parentTemp.append(parent[ii])
        
        parentList.append(parentTemp)

    return parentList                        