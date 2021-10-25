# tian_IPCMB
import numpy as np
from rpi_d3m_primitives.featSelect.tian_RecognizePC_faster import RecognizePC
#from rpi_d3m_primitives.featSelect.tian_checkDataSize import checkDataSize
from rpi_d3m_primitives.featSelect.helperFunctions import joint
from rpi_d3m_primitives.featSelect.conditionalMI import cmi

def tian_IPCMB (train_data, target, threshold):  #train_data is not including targets, targets is the label vector
    NumTest = 0
    numSample = train_data.shape[0]
    numf = train_data.shape[1] # do not include the target 
    CanMB = np.arange(numf) 
    #target = target.reshape([numSample,1])
    Results = RecognizePC(target, CanMB, train_data, threshold, NumTest)
    
    PC = Results[0]
    Sepset_t = Results[1]
    NumTest = Results[2]
    #cutSetSize = Results[3]
    
    MB = PC
    #association = []
    
    #Recognize a true positive, and its PC as spouse candidate
    children = []
    targetindex = 0

    for xind in np.arange(len(PC)):
        X = PC[xind]
        CanADJX = np.arange(numf)
        rest_idx = np.setdiff1d(np.arange(numf), X) #numf-1
        temp_trainD = np.hstack((target, train_data[:, rest_idx]))
        Results = RecognizePC(train_data[:,X], CanADJX, temp_trainD, threshold, NumTest)
        temp_CanSP = Results[0]
        NumTest = Results[2]
                
        if ~np.in1d(targetindex, temp_CanSP):
            MB = np.setdiff1d(MB, X)
            continue
        
        temp_idx = np.where(temp_CanSP != 0)
        CanSP = temp_CanSP[temp_idx]
        temp_idx = np.where(CanSP <= X)
        CanSP[temp_idx] = CanSP[temp_idx] - 1
        
        # recognize true positives
        DiffY = np.setdiff1d(CanSP, MB)  # in CanSP but not in MB 
        DiffY = np.setdiff1d(DiffY, X)   # X should not in Sepset
        
        for yind in np.arange(len(DiffY)):
            Y = DiffY[yind]
            SepsetTY = Sepset_t[Y]
            cmbVector = joint(train_data[:, list(set(SepsetTY).union(set([X])))])
            NumTest = NumTest + 1
            if cmi(train_data[:,Y], target, cmbVector,0) > threshold:
                children = set(children).union(set([X]))
                children = list(children)
                MB = set(MB).union(set([Y]))
                MB = list(MB)
    
    
    result = []
    result.append(np.array(MB))
    result.append(PC)
    result.append(NumTest)
    result.append(children)
    
    return result