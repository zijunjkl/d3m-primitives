#%% keyi_STMB_adaptive_soft.m
import numpy as np
from rpi_d3m_primitives.featSelect.Keyi_RecognizePC_adaptive_soft import find_PC_adpative
from rpi_d3m_primitives.featSelect.Keyi_checkDatasize import Keyi_checkDataSize
from rpi_d3m_primitives.featSelect.helperFunctions import joint
from rpi_d3m_primitives.featSelect.adaptiveMI import CMI_adaptive_pure_soft

def Keyi_STMB_Adaptive_soft(train_data, targets, threshold = 0.02):
    NumTest = 0
    hm_HypoTest = 0
    numf = train_data.shape[1]  # feature number
    
    # %% Recognize Target PC
    CanMB = np.arange(numf)    # candidates
    
    data_check = 1
    PCD, Sepset_t, NumTest, cutSetSize, hm_HypoTest = find_PC_adpative(targets, CanMB, train_data, threshold, NumTest, hm_HypoTest)
    
    spouse = [[]]*numf
    #scores = []
    Des = [[]]*PCD.size
    datasizeFlag = 1
    #%% Find Markov blanket
    
    for yind in range(0, PCD.size):
        flag = 0
        y = PCD[yind]
        searchset = np.setdiff1d(CanMB, PCD)
        
        for xind in range(0, searchset.size):
            x = searchset[xind]
            col = set(Sepset_t[x]).union(set([y]))
            cmbVector = joint(train_data[:, np.array(list(col))])
            if data_check == 1:
                datasizeFlag = Keyi_checkDataSize(train_data[:,x], targets, cmbVector)            
            if datasizeFlag != 1:
                NumTest = NumTest + 1
                cond_data = train_data[:, np.array(list(col))]
                cond_mi, hm_HypoTest = CMI_adaptive_pure_soft(train_data[:,x], targets, cond_data, hm_HypoTest)
                if cond_mi > threshold:                    # v structure             
                    temp = set(PCD).union(set([x]))
                    for s in np.setdiff1d(np.array(list(temp)), y): 
                        cond_data = train_data[:,s]
                        cond_mi, hm_HypoTest = CMI_adaptive_pure_soft(train_data[:,y], targets, cond_data, hm_HypoTest) 
                        if cond_mi < threshold:
                            temp = set(Des[yind]).union(set([y]))
                            Des[yind] = np.array(list(temp))
                            flag = 1; 
                            break
                        else:
                            temp = set(spouse[y]).union(set([x]))
                            spouse[y]= np.array(list(temp))

            if flag == 1:                            
               break
    
    PCD = np.setdiff1d(PCD, Des[:])

    #%% Shrink spouse
    NonS = []
    S = []

    for i in np.setdiff1d(np.arange(numf), PCD):
        spouse[i] = []   # empty                                     

    for y in np.arange(len(spouse)):
        if spouse[y] != []:
           S.append( y)    # Y has spouses
           # shrink
           spousecan = spouse[y]
           for sind in np.arange(spousecan.size):
               s = spousecan[sind]
               col = set([y]).union(set(spousecan),set(PCD))
               cmbVector = joint(train_data[:, np.setdiff1d(np.array(list(col)), s)])
               if data_check == 1:
                   datasizeFlag = 0
                   datasizeFlag = Keyi_checkDataSize(train_data[:,s], targets, cmbVector)
               if datasizeFlag != 1:
                  NumTest = NumTest + 1
                  cond_data = train_data[:, np.setdiff1d(np.array(list(col)), s)]
                  cond_mi, hm_HypoTest = CMI_adaptive_pure_soft(train_data[:,s], targets, cond_data, hm_HypoTest)
                  if cond_mi < threshold:
                     NonS = set(NonS).union(set([s]))
           spouse[y] = np.setdiff1d(spousecan, np.array(list(NonS)))
           NonS = []
                                                            
    b = [];
    for i in np.arange(len(spouse)):
         if spouse[i] != []:
             b = set(b).union(set(spouse[i]))
    # remove false spouse from PC
    M = PCD       # setdiff(PCD,S); % M has no spouses in PCD set
    PCsize = M.size
    testSet = set(S).union(set(b))
    #testSet = np.array(list(temp))
    C = np.zeros(shape = (PCsize, 1))
    for x in M:
       col = set(PCD).union(set(testSet))
       cmbVector = joint(train_data[:, np.setdiff1d(np.array(list(col)), x)])
       if data_check == 1:
           datasizeFlag = 0
           datasizeFlag = Keyi_checkDataSize(train_data[:, x], targets, cmbVector)
       if datasizeFlag != 1:
            NumTest = NumTest + 1
            cond_data = train_data[:, np.setdiff1d(np.array(list(col)), x)]
            cond_mi, hm_HypoTest = CMI_adaptive_pure_soft(train_data[:,x], targets, cond_data, hm_HypoTest) 
            if cond_mi < threshold:
               PCD = np.setdiff1d(PCD, x)
            
            datasizeFlag = 0
     
    PCsize2 =np.mean(C)
    MB = set(PCD).union(set(b))
    
    result = []
    result.append(np.array(list(MB)))
    result.append(PCD)
    result.append(spouse)
    result.append(NumTest)
    result.append(cutSetSize)
    result.append(PCsize)
    result.append(PCsize2)
    
    return result                                                              