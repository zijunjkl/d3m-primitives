import numpy as np
from itertools import combinations
from rpi_d3m_primitives.featSelect.HypothesisTest import GTest_CI
from rpi_d3m_primitives.featSelect.HypothesisTest import GTest_I
from rpi_d3m_primitives.featSelect.mutualInformation import mi
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.featSelect.tian_checkDataSize import checkDataSize
from rpi_d3m_primitives.featSelect.helperFunctions import joint

def RecognizePC_Gtest(targets, ADJt, data):
    NonPC = []
    cutSetSize = 0
    data_check = 0
    Sepset = [[]]*data.shape[1]
    #% Search
    datasizeFlag = 0
    while ADJt.size > cutSetSize:
        for xind in range(0, ADJt.size):        # for each x in ADJt
            X = ADJt[xind]
            if cutSetSize == 0:
                Independency = GTest_I(data[:,X], targets)   
                if Independency == 1:
                      NonPC.append(X)              
            elif cutSetSize == 1: 
                Diffx = np.setdiff1d(ADJt, X)
                C = list(combinations(Diffx, cutSetSize))
                for sind in range(0, len(C)):                    # for each S in ADJT\x, size
                        S = np.array(list(C[sind]))
                        cmbVector = joint(data[:, S])
                        if data_check:
                            datasizeFlag = checkDataSize(data[:, X], targets, cmbVector)
                        if datasizeFlag != 1:
                            Independency = GTest_CI(data[:,X], targets, cmbVector)                    
                            if Independency == 1:
                                NonPC = set(NonPC).union(set([X]))
                                Sepset[X] = set(Sepset[X]).union(set(S))
                                break
                        else:
                            break
            else:                                # set size > 1
                Diffx = np.setdiff1d(ADJt, X)                
                C = list(combinations(Diffx, cutSetSize - 1))
                midBreakflag = 0
                for sind in range(0, len(C)):             # for each S in ADJT\x, size
                    S = np.array(list(C[sind]))
                    RestSet = np.setdiff1d(Diffx, S); 
                    for addind in range(0, RestSet.size):
                        col = set(S).union(set([RestSet[addind]]))
                        cmbVector = joint(data[:, np.array(list(col))])
                        if data_check:
                            datasizeFlag = checkDataSize(data[:, X], targets, cmbVector)
                        if datasizeFlag != 1:
                            Independency = GTest_CI(data[:,X], targets, cmbVector)
                            if Independency == 1:
                                NonPC = set(NonPC).union(set([X]))
                                # Line has an error
                                Sepset[X] = set(Sepset[X]).union(set(S),set([RestSet[addind]]))
                                midBreakflag = 1
                                break                                                    
                        else:
                            break
                    if midBreakflag == 1:
                        break
        if len(NonPC) > 0:
           ADJt = np.setdiff1d(ADJt, np.array(list(NonPC)))
           cutSetSize = cutSetSize + 1
           NonPC = []
        elif datasizeFlag == 1:
           break
        else:
           cutSetSize = cutSetSize + 1
    
    ADJ = ADJt
    
    result = []
    result.append(ADJ)
    result.append(Sepset)
    result.append(cutSetSize)
    
    return result