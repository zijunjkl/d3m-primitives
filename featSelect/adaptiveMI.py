import numpy as np
from rpi_d3m_primitives.featSelect.mutualInformation import mi
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.featSelect.adaptivePartitioning import jointPDFAdapPartition 
from rpi_d3m_primitives.featSelect.helperFunctions import joint

def get_MI_from_joint_distribution(joint_dist):
    m,n = joint_dist.shape
    px_r = np.sum(joint_dist, 1)
    py_r = np.sum(joint_dist, 0)
    
    mutual_info = 0
    
    for i in range(m):
        for j in range(n):
            if px_r[i] != 0 and py_r[j] != 0 and joint_dist[i,j] != 0:
                mutual_info = mutual_info + joint_dist[i,j]*np.log2(joint_dist[i,j]/px_r[i]/py_r[j])
            
    return mutual_info

def MI_adaptive_soft(X, Y, hm_HypoTest):
    
    Cx,X = np.unique(X, return_inverse = True)
    Cy,Y = np.unique(Y, return_inverse = True)
    
    X = X + 1
    Y = Y + 1
    
    m = len(Cx)
    n = len(Cy)
        
    joint_dist = np.ones([m,n]) * (1/(m*n))
    mutual_info = 0
    _,a = np.unique((np.column_stack((X, Y))),axis=0,return_counts=True)
    if min(a) < 3:
        mutual_info = mi(X,Y)
        if m == 1 or n == 1:
            mutual_info = np.inf
            result = []
            result.append(mutual_info)
            result.append(joint_dist)
            result.append(hm_HypoTest)
            return result
        
        result = []
        result.append(mutual_info)
        result.append(joint_dist)
        result.append(hm_HypoTest)
        return result
    
    if min(a) > 10:
        mutual_info = mi(X,Y)
        if m == 1 or n == 1:
            mutual_info = 0
            result = []
            result.append(mutual_info)
            result.append(joint_dist)
            result.append(hm_HypoTest)
            return result
        
        result = []
        result.append(mutual_info)
        result.append(joint_dist)
        result.append(hm_HypoTest)
        return result
    
    if m == 1 or n == 1:
        mutual_info = np.inf
        #result = []
        #result.append(mutual_info)
        #result.append(joint_dist)
        #result.append(hm_HypoTest)
        return mutual_info, joint_dist, hm_HypoTest
    #Estimate the joint probability mass by adaptive partitioning
    joint_dist, hm_HypoTest, isUniform = jointPDFAdapPartition(X, Y, m, n, hm_HypoTest)
    
    mutual_info = np.abs(get_MI_from_joint_distribution(joint_dist))
    if isUniform or mutual_info <= 0.0000001:
        mutual_info = np.inf
    return mutual_info, joint_dist, hm_HypoTest


def CMI_adaptive_pure_soft(X, Y, cond_set, hm_HypoTest):
    cond_mi = 0
    if (len(cond_set.shape) == 1):
        cond_set = cond_set.reshape((cond_set.size,1))
    if (cond_set.size == 0):
        results = MI_adaptive_soft(X, Y, hm_HypoTest)
        cond_mi = results[0]
        hm_HypoTest = results[1]
        
        results = []
        results.append(cond_mi)
        results.append(hm_HypoTest)
        return results
    
    naive_cmi = cmi(X, Y, cond_set)
    if naive_cmi == 0:
        results = []
        results.append(cond_mi)
        results.append(hm_HypoTest)
        return results
    
    Cx, X = np.unique(X, return_inverse = True)
    Cy, Y = np.unique(Y, return_inverse = True)
    
    m = len(Cx)
    n = len(Cy)
    
    hm_sample, hm_condvar = cond_set.shape
    entire_uniform = 1
    
    if hm_condvar == 1:
        combo_set = np.unique(cond_set)
        j = []
        for i in range(combo_set.shape[0]):
            pattern = combo_set[i]
            sub_cond_idx = np.argwhere(cond_set==pattern).T[0]
            p_cond = len(sub_cond_idx) / hm_sample
            
            sub_cond_idx = np.array(sub_cond_idx)
            results = MI_adaptive_soft(X[sub_cond_idx], Y[sub_cond_idx], hm_HypoTest)
            temp_mi = results[0]
            hm_HypoTest = results[2]
            
            if temp_mi == np.inf:
                temp_mi = 0
            else:
                entire_uniform = 0
            cond_mi = cond_mi + p_cond*temp_mi
    else:
        var_1 = cond_set[:,1]
        var_2 = joint(cond_set[:, 2:])
        
        C1,var_1 = np.unique(X, return_inverse = True)
        C2,var_2 = np.unique(Y, return_inverse = True)
        #C1 = np.unique(X, return_inverse = True)
        #C2 = np.unique(Y, return_inverse = True)
        
        p = len(C1)
        q = len(C2)
        
        joint_set, hm_HypoTest, isUniform = jointPDFAdapPartition(var_1, var_2, p, q, hm_HypoTest)
        
        for j in range(p):
            for k in range(q):
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                index = get_indexes(C1[j], var_1)
                index = np.array(index)
                sub_cond_idx = get_indexes(C2[k], var_2[index])
                sub_cond_idx = np.array(sub_cond_idx)
                sub_cond_idx = sub_cond_idx.astype(int)
                p_cond = len(sub_cond_idx) / hm_sample
                if len(sub_cond_idx) == 0:
                    temp_mi = 0
                else:
                    results = MI_adaptive_soft(X[sub_cond_idx], Y[sub_cond_idx], hm_HypoTest)
                    temp_mi = results[0]
                    hm_HypoTest = results[2]
                if temp_mi == np.inf:
                    temp_mi = 0
                else:
                    entire_uniform = 0
                
                cond_mi = cond_mi + p_cond*temp_mi
    
    if entire_uniform:
        cond_mi = np.inf
    return cond_mi, hm_HypoTest