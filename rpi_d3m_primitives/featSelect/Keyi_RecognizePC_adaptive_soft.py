import numpy as np
from itertools import combinations
from rpi_d3m_primitives.featSelect.adaptiveMI import MI_adaptive_soft,CMI_adaptive_pure_soft
from rpi_d3m_primitives.featSelect.Keyi_checkDatasize import Keyi_checkDataSize
from rpi_d3m_primitives.featSelect.helperFunctions import joint

# author: Keyi
def find_PC_adpative(targets, ADJt, data, THRESHOLD, NumTest, hm_HypoTest):
    MIs = []
    CMIs = []
    NonPC = []
    cutSetSize = 0
    data_check = 1
    #targets = data[:, T]
    Sepset = [[]]*data.shape[1]
    #% Search
    datasizeFlag = 0
    while ADJt.size > cutSetSize:
        for xind in range(0, ADJt.size):        # for each x in ADJt
            X = ADJt[xind]
            if cutSetSize == 0:
                NumTest = NumTest + 1
                marg_mi,_,hm_HypoTest = MI_adaptive_soft(data[:,X], targets, hm_HypoTest)
                MIs.append([marg_mi])   #compute mutual information           
                if marg_mi <= THRESHOLD:
                     NonPC.append(X)              
            elif cutSetSize == 1: 
                Diffx = np.setdiff1d(ADJt, X)
                C = list(combinations(Diffx, cutSetSize))
                for sind in range(0, len(C)):                    # for each S in ADJT\x, size
                        S = np.array(list(C[sind]))
                        cmbVector = joint(data[:, S])
                        if data_check:
                            datasizeFlag = Keyi_checkDataSize(data[:, X], targets, cmbVector)
                        if datasizeFlag != 1:
                            NumTest = NumTest + 1
                            cond_data = data[:,S]
                            cond_mi, hm_HypoTest = CMI_adaptive_pure_soft(data[:,X], targets, cond_data, hm_HypoTest)
                            CMIs.append([cond_mi])
                            if cond_mi <= THRESHOLD:
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
                            datasizeFlag = Keyi_checkDataSize(data[:, X], targets, cmbVector)
                        if datasizeFlag != 1:
                            NumTest = NumTest + 1
                            cond_data = data[:, np.array(list(col))]
                            cond_mi,hm_HypoTest = CMI_adaptive_pure_soft(data[:,X], targets, cond_data, hm_HypoTest)
                            CMIs.append([cond_mi])
                            if cond_mi <= THRESHOLD:
                                NonPC = set(NonPC).union(set([X]))
                                Sepset[X] = set(Sepset[X]).union(set(S),set([RestSet[addind]]))
                                midBreakflag = 1
                                break                                                    
                        else:
                            break
                    if midBreakflag == 1:
                        break

        if len(NonPC) > 0:
           ADJt = np.setdiff1d(ADJt, NonPC)
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
    result.append(NumTest)
    result.append(cutSetSize)
    result.append(MIs)
    
    return result
