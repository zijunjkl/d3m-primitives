import numpy as np
from rpi_d3m_primitives.featSelect.combineJointDist import combine_joint_dist_table
from rpi_d3m_primitives.featSelect.statsFunctions import get_num_value_HypoTest, getPvalue_Chisquare, getSquaredError

def jointPDFAdapPartition(X,Y,num_state_x, num_state_y, hm_HypoTest = 0):
    hm_recur = 0
    isUniform = 0
    hm_samples = len(X)
    
    m = num_state_x
    n = num_state_y
    
    joint_dist = np.zeros([m,n])
    
    part_X = [1, m]
    part_Y = [1, n]
    
    part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X, Y, m, n, hm_HypoTest, hm_recur)

    joint_state_counts = np.zeros(num_state_x*num_state_y)
    joint_states = np.column_stack((X-1,Y-1))
    joint_indices,joint_counts = np.unique(joint_states,axis=0, return_counts=True)
    joint_indices = joint_indices.T[0]*n + joint_indices.T[1]
    joint_state_counts[joint_indices] = joint_counts
    joint_state_counts /= hm_samples
    joint_dist = np.reshape(joint_state_counts,(num_state_x,num_state_y))
    for i in range(1,len(part_X)):
        curr_sx = part_X[i]-1
        prev_sx = part_X[i-1]-1
        for j in range(1,len(part_Y)):
            curr_sy = part_Y[j]-1
            prev_sy = part_Y[j-1]-1
            x_range = curr_sx - prev_sx
            y_range = curr_sy - prev_sy
            if (x_range<=1 and y_range<=1):
                continue
            partition_mass = sum(joint_dist[prev_sx:curr_sx,prev_sy:curr_sy])
            partition = np.full((x_range,y_range),partition_mass/(x_range*y_range))
            joint_dist[prev_sx:curr_sx,prev_sy:curr_sy] = partition

    if hm_recur == 1:
        isUniform = 1
    else:
        if m != 1 and n!= 1:
            combined = combine_joint_dist_table(joint_dist, X, Y, part_X, part_Y, hm_HypoTest)
            joint_dist, hm_HypoTest = combined[0],combined[1]
    
    #results = []
    #results.append(joint_dist)
    #results.append(hm_HypoTest)
    #results.append(isUniform)
    return joint_dist, hm_HypoTest, isUniform

def recurrAdapPartition(part_X, part_Y, X, Y, m, n, hm_HypoTest, hm_recur, sig_level= 0.15):
    
    hm_recur += 1
    
    if len(X) == 0 or len(Y) == 0:
        results = []
        results.append(part_X)
        results.append(part_Y)
        results.append(hm_HypoTest)
        results.append(hm_recur)
        return results
    
    flag_x = 1
    flag_y = 1
    
    if len(part_Y) == n+1:
        flag_y = 0
    
    if len(part_X) == m+1:
        flag_x = 0
    
    if flag_y == 0 and flag_x == 0:
        results = []
        results.append(part_X)
        results.append(part_Y)
        results.append(hm_HypoTest)
        results.append(hm_recur)
        return results
    
    hm_x, hm_y = get_num_value_HypoTest(X, Y, part_X, part_Y)
    p_val = getPvalue_Chisquare(X, Y, hm_x, hm_y)
    
    if p_val > sig_level:
        #hm_HypoTest = hm_HypoTest + 1
        results = []
        results.append(part_X)
        results.append(part_Y)
        results.append(hm_HypoTest)
        results.append(hm_recur)
        return results
    
    if flag_x == 0 and flag_y:
        sy = getSquaredError(Y, part_Y)[0]
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        flag = get_indexes(sy, part_Y) 
        if (sy < n and not flag) or (sy == n and len(flag) == 1):            
            part_Y.append(sy)
            part_Y = sorted(part_Y)
        else:
            results = []
            results.append(part_X)
            results.append(part_Y)
            results.append(hm_HypoTest)
            results.append(hm_recur)
            return results
        
        bins = np.digitize(Y, [1, sy, n+1])
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        index_1 = get_indexes(1, bins)
        index_2 = get_indexes(2, bins)
        
        if len(index_1) == 0: # if empty
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
        else:
            index_1 = np.array(index_1)
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_1], Y[index_1], m, n, hm_HypoTest, hm_recur)
        
        if len(index_2) == 0:
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
        else:
            index_2 = np.array(index_2)
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_2], Y[index_2], m, n, hm_HypoTest, hm_recur)
            
    elif flag_y == 0 and flag_x:
        sx = getSquaredError(X, part_X)[0]
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        flag = get_indexes(sx, part_X)
        if (sx < m and not flag) or (sx == m and len(flag) == 1):
            part_X.append(sx)
            part_X = sorted(part_X)
        else:
            results = []
            results.append(part_X)
            results.append(part_Y)
            results.append(hm_HypoTest)
            results.append(hm_recur)
            return results
        
        bins = np.digitize(X, [1, sx, m+1])
        index_1 = get_indexes(1, bins)
        index_2 = get_indexes(2, bins)
        
        if len(index_1) == 0: # if empty
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
        else:
            index_1 = np.array(index_1)
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_1], Y[index_1], m, n, hm_HypoTest, hm_recur)
        
        if len(index_2) == 0:
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
        else:
            index_2 = np.array(index_2)
            part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_2], Y[index_2], m, n, hm_HypoTest, hm_recur)
        
    else:
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        sx,tx = getSquaredError(X, part_X)
        sy,ty = getSquaredError(Y, part_Y)
        
        if tx > ty:
            flag = get_indexes(sx, part_X)
            if (sx < m and not flag) or (sx == m and len(flag) == 1):
                part_X.append(sx)
                part_X = sorted(part_X)
            else:
                results = []
                results.append(part_X)
                results.append(part_Y)
                results.append(hm_HypoTest)
                results.append(hm_recur)
                return results
            
            temp = [1, sx, m+1]
            temp.sort()
            temp = np.asarray(temp)
            bins = np.digitize(X, temp)
            index_1 = get_indexes(1, bins)
            index_2 = get_indexes(2, bins)
            
            if len(index_1) == 0: # if empty
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
            else:
                index_1 = np.array(index_1)
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_1], Y[index_1], m, n, hm_HypoTest, hm_recur)
            
            if len(index_2) == 0:
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
            else:
                index_2 = np.array(index_2)
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_2], Y[index_2], m, n, hm_HypoTest, hm_recur)
                
        else:
            flag = get_indexes(sy, part_Y)
            if (sy < n and not flag) or (sy == n and len(flag) == 1):
                part_Y.append(sy)
                part_Y = sorted(part_Y)
            else:
                results = []
                results.append(part_X)
                results.append(part_Y)
                results.append(hm_HypoTest)
                results.append(hm_recur)
                return results
            
            bins = np.digitize(Y, [1, sy, n+1])
            index_1 = get_indexes(1, bins)
            index_2 = get_indexes(2, bins)
            if len(index_1) == 0: # if empty
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
            else:
                index_1 = np.array(index_1)
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_1], Y[index_1], m, n, hm_HypoTest, hm_recur)
            
            if len(index_2) == 0:
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, [], [], m, n, hm_HypoTest, hm_recur)
            else:
                index_2 = np.array(index_2)
                part_X, part_Y, hm_HypoTest, hm_recur = recurrAdapPartition(part_X, part_Y, X[index_2], Y[index_2], m, n, hm_HypoTest, hm_recur)
            
            
        
    results = []
    results.append(part_X)
    results.append(part_Y)
    results.append(hm_HypoTest)
    results.append(hm_recur)
    return results