import numpy as np
from scipy.stats import chi2
from itertools import combinations
from rpi_d3m_primitives.featSelect.mutualInformation import mi
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.featSelect.helperFunctions import joint

def GTest_I(X, Y):
    sig_level_indep = 0.05
    hm_x = len(np.unique(X))
    hm_y = len(np.unique(Y))
    
    hm_samples = X.size
    g = 2*hm_samples*mi(X,Y,0)
    
    p_val  = 1 - chi2.cdf(g, (hm_x-1)*(hm_y-1))
    
    if p_val < sig_level_indep:
        Independency = 0  # reject the Null-hypothesis
    else:
        Independency = 1
        
    return Independency

def GTest_CI(X,Y,Z):
    g = 0
    sig_level_indep = 0.05
    
    hm_x = len(np.unique(X))
    hm_y = len(np.unique(Y))
    hm_z = len(np.unique(Z))
    
    hm_samples = X.size
    
    if Z.size == 0:
        return GTest_I(X,Y)
    else:
#        if (len(Z.shape)>1 and Z.shape[1]>1):
#            Z = joint(Z)
#        states = np.unique(Z)
#        for i in states:
#            pattern = i
#            sub_cond_idx = np.where(Z == pattern)
#            temp_mi = mi(X[sub_cond_idx], Y[sub_cond_idx],0)
#            g = g + sub_cond_idx.length*temp_mi
        g = 2*hm_samples*cmi(X,Y,Z)
        p_val = 1 - chi2.cdf(g, (hm_x-1)*(hm_y-1)*hm_z)
        
        if p_val < sig_level_indep:
            Independency = 0  # reject the Null-hypothesis
        else:
            Independency = 1
        
    return Independency