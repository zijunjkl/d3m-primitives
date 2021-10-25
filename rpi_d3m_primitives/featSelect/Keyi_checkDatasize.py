import numpy as np

def Keyi_checkDataSize(X, T, Z):
    datasizeFlag = 0
    alpha = 5
    
    Xcard = len(np.unique(X))
    Tcard = len(np.unique(T))
    
    Zcard = np.unique(Z)
    
    a = np.histogram(Z, Zcard)
    if (len(a[0]) == 0):
    	datasizeFlag = 0
    elif np.min(a[0]) < alpha*Xcard*Tcard:
        datasizeFlag = 1
    
    return datasizeFlag