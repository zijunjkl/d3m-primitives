import numpy as np
from rpi_d3m_primitives.featSelect.RecognizePC_Gtest import RecognizePC_Gtest
from rpi_d3m_primitives.featSelect.tian_checkDataSize import checkDataSize
from rpi_d3m_primitives.featSelect.helperFunctions import joint
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.featSelect.HypothesisTest import GTest_CI


def STMB_G_test(train_data, targets): 
    numf = train_data.shape[1]  # feature number
    #targets = data[:, targetindex]    # selected index data 
    # %% Recognize Target PC
    CanMB = np.arange(numf)    # candidates
    
    Results = RecognizePC_Gtest(targets, CanMB, train_data)
    PCD = Results[0]
    Sepset_t = Results[1]
    cutSetSize = Results[2]
    spouse = [[]]*numf
    #print("===========PC Result==========")
    #print(PCD)
    # print(Sepset_t)
    # print(cutSetSize)
    #scores = []
    Des = [[]]*PCD.size
    datasizeFlag = 0
    #%% Find Markov blanket
    for yind in range(PCD.size):
        flag = 0
        y = PCD[yind]
        searchset = np.setdiff1d(CanMB, PCD)
        
        for xind in range(searchset.size):
            x = searchset[xind]
            col = set(Sepset_t[x]).union(set([y]))
            cmbVector = joint(train_data[:, np.array(list(col))])
#            datasizeFlag = checkDataSize(train_data[:, x], targets, cmbVector)
            #print("datasizeFlag",x,datasizeFlag)
            if datasizeFlag != 1:
                Independency = GTest_CI(train_data[:,x], targets, cmbVector)          
                if Independency == 0: # V structure
                    for s in np.setdiff1d(np.union1d(PCD,[x]), np.array([y])): 
                        Independency = GTest_CI(train_data[:,y], targets, train_data[:,s])
                        if Independency == 1:
                            temp = set(Des[yind]).union(set([y]))
                            Des[yind] = np.array(list(temp))
                            flag = 1
                            break
                        else:
                            temp = set(spouse[y]).union(set([x]))
                            spouse[y]= np.array(list(temp))

            if flag == 1:                            
               break
    
    des = [item for sublist in Des for item in sublist]
    PCD = np.setdiff1d(PCD, des)
    #print(PCD)
    #assert(1==2)
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
#               datasizeFlag = checkDataSize(train_data[:, s], targets, cmbVector)
               if datasizeFlag != 1:
                  Independency = GTest_CI(train_data[:,s], targets, cmbVector)
                  if Independency == 1:
                     NonS = set(NonS).union(set([s]))
           spouse[y] = np.setdiff1d(spousecan, np.array(list(NonS)))
           NonS = []
                                                            
    b = []
    for i in range(len(spouse)):
        if len(spouse[i]) > 0:
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
#       datasizeFlag = checkDataSize(train_data[:, x], targets, cmbVector)
       if datasizeFlag != 1:
            Independency = GTest_CI(train_data[:,x], targets, cmbVector)
            if Independency == 1:
               PCD = np.setdiff1d(PCD, x)                                                                      
    PCsize2 =np.mean(C)
    MB = set(PCD).union(set(b))
    
    result = []
    result.append(np.array(list(MB)))
    result.append(PCD)
    result.append(spouse)
    result.append(Sepset_t)
    result.append(cutSetSize)
    result.append(PCsize)
    result.append(PCsize2)
    
    return result