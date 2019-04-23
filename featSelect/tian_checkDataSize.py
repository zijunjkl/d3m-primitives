import numpy as np

def checkDataSize(X, T, S):
    if (S.size == 0):
        return 1
    # check enough data is valid for independence tests
    # at time 5 times the degree of freedom
    datasizeFlag = 0
    alpha = 2

    Xcard = np.unique(X).size
    Tcard = np.unique(T).size

    # check data size
    temp = np.unique(S)
    Scard = list(temp)
    Scard.append(np.max(temp)+1)
    Scard = np.array(Scard)
    
    [a, b] = np.histogram(S, Scard)
    #% all has to be fit data
    if min(a) < alpha * Xcard * Tcard:
        datasizeFlag = 1
    return datasizeFlag