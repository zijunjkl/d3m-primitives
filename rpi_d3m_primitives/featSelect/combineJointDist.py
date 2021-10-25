from rpi_d3m_primitives.featSelect.statsFunctions import get_num_value_HypoTest,getPvalue_Chisquare,getSquaredError
import numpy as np


def combine_joint_dist_table(joint_dist, X, Y, part_X, part_Y, hm_HypoTest, sig_level=0.15):
    
    flag_vertical = 0
    hm_sx = len(part_X) - 1
    hm_sy = len(part_Y) - 1
    joint_table = joint_dist
    for i in range(hm_sx):
        if hm_sx == max(part_X):
            break
        if part_X[i+1] == max(part_X) and part_X[-2] != max(part_X):
            Len = part_X[i+1] - part_X[i] + 1
            prob = np.zeros([Len, 1])
            space_x = np.arange(part_X[i], part_X[i+1]+1)
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y >= x]
            num_sampx = get_indexes(part_X[i], X)
            num_sampx = np.array(num_sampx)
            for j in range(max(part_Y)):
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                num_sampy = get_indexes(j+1, Y)
                num_sampy = np.array(num_sampy)
                joint_idx = np.intersect1d(num_sampx, num_sampy)
                hm_x, hm_y = get_num_value_HypoTest(X, Y, part_X, part_Y)
                joint_idx = joint_idx.astype(int)
                p_val = getPvalue_Chisquare(X[joint_idx], Y[joint_idx], hm_x, 1)
                
                if p_val > sig_level:
                    flag_vertical = 1
                    continue
                else:
                    #hm_HypoTest = hm_HypoTest - 1
                    prob = np.zeros((1,Len))
                    for k in range(Len):
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                        temp = get_indexes(part_X[i]+k, X[joint_idx])
                        prob[0,k] = len(temp)/len(X)
                    
                    rem_prob = 1 - sum(prob[0])
                    if rem_prob != 0 :
                        joint_table[space_x-1, j] = 0
                        joint_table = rem_prob / sum(joint_table) * joint_table
                        joint_table[space_x-1, j] = prob
                        flag_vertical = 1
            continue
        
        if part_X[i+1] - part_X[i] >= 2:
            Len = part_X[i+1] - part_X[i]
            prob = np.zeros([Len, 1])
            space_x = np.arange(part_X[i], part_X[i+1]+1)
            flat_X = X.flatten()
            num_sampx = np.argwhere((flat_X>=part_X[i]) & (flat_X<part_X[i+1])).flatten()
            space_x = space_x[:-1]
            
            for j in range(max(part_Y)):
                num_sampy = np.argwhere(Y.flatten()==j+1).flatten()
                joint_idx = np.intersect1d(num_sampx, num_sampy)
                hm_x, hm_y = get_num_value_HypoTest(X, Y, part_X, part_Y)
                p_val = getPvalue_Chisquare(X[joint_idx], Y[joint_idx], hm_x, 1)
                if p_val > sig_level:
                    flag_vertical = 1
                    continue
                else:
                    #hm_HypoTest = hm_HypoTest - 1
                    prob = np.zeros((1,Len))
                    flat_joint = X[joint_idx].flatten()
                    for k in range(Len):
                        temp = np.argwhere(flat_joint == part_X[i] + k).flatten()
                        prob[0,k] = len(temp)/len(X)
                    rem_prob = 1 - sum(prob[0])
                    if (rem_prob != 0 and any(np.isnan(sum(joint_table))) and any(sum(joint_table)==0)):
                        joint_table[space_x-1, j] = 0
                        joint_table = rem_prob/sum(joint_table)*joint_table
                        joint_table[space_x-1, j] = prob
                        flag_vertical = 1
        
    #Horizontal search
    if flag_vertical :
        results = []
        results.append(joint_table)
        results.append(hm_HypoTest)
        results.append(joint_table)
        results.append(hm_HypoTest)
        return results
    
    for i in range(hm_sy):
        if flag_vertical == 1:
            break
        if part_Y[i+1] == max(part_Y) and part_Y[-2] != max(part_Y):        
            Len = part_Y[i+1] - part_Y[i] + 1
            prob = np.zeros([1, Len])
            space_y = np.arange(part_Y[i], part_Y[i+1]+1)
            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y >= x]
            num_sampy = get_indexes(part_Y[i], Y)
            num_sampy = np.array(num_sampy)
            
            for j in range(max(part_X)):
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                num_sampx = get_indexes(j+1, X)
                num_sampx = np.array(num_sampx)
                joint_idx = np.intersect1d(num_sampx, num_sampy)
                hm_x, hm_y = get_num_value_HypoTest(X,Y,part_X, part_Y)
                p_val = getPvalue_Chisquare(X[joint_idx], Y[joint_idx], 1, hm_y)
                if p_val > sig_level:
                    continue
                else:
                    #hm_HypoTest = hm_HypoTest - 1
                    prob = np.zeros((1,Len))
                    for k in range(Len):
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                        temp = get_indexes(part_Y[i] + k, Y[joint_idx])
                        prob[0,k] = len(temp)/len(X)
                    rem_prob = 1 - sum(prob[0])
                    if rem_prob != 0:
                        joint_table[j, space_y-1] = 0
                        joint_table = rem_prob / sum(joint_table) * joint_table
                        joint_table[j, space_y-1] = prob
            
            continue
        
        if part_Y[i+1] - part_Y[i] >= 2:
            Len = part_Y[i+1] - part_Y[i]
            prob = np.zeros(1, Len)
            space_y = np.arange(part_Y[i], part_Y[i+1]+1)
            get_indexes = lambda x1, x2, xs: [i for (y, i) in zip(xs, range(len(xs))) if y >= x1 and y < x2]
            num_sampy = get_indexes(part_Y[i], part_Y[i+1], Y)
            num_sampy = np.array(num_sampy)
            space_y = space_y[:-1]
            
            for j in range(max(part_X)):
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                num_sampx = get_indexes(j+1, X)
                num_sampx = np.array(num_sampx)
                joint_idx = np.intersect1d(num_sampx, num_sampy)
                hm_x, hm_y = get_num_value_HypoTest(X,Y,part_X, part_Y)
                p_val = getPvalue_Chisquare(X[joint_idx], Y[joint_idx], 1, hm_y) 
                if p_val > sig_level:
                    continue
                else:
                    #hm_HypoTest = hm_HypoTest - 1
                    prob = np.zeros((1,Len))
                    for k in range(Len):
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if y == x]
                        temp = get_indexes(part_Y[i] + k, Y[joint_idx])
                        prob[0,k] = len(temp)/len(X)
                    rem_prob = 1 - sum(prob[0])
                    if rem_prob != 0:
                        joint_table[j, space_y-1] = 0
                        joint_table = rem_prob / sum(joint_table) * joint_table
                        joint_table[j, space_y-1] = prob   
    
    m = max(part_X)
    n = max(part_Y)
    flag_vertical = 0
    
    for i in range(m):
        index = np.argwhere(X==i+1).T[0]
        p_val = getPvalue_Chisquare(X[index], Y[index], 1, n)
        if p_val > sig_level:
            total_prob = len(index)/len(X)
            rem_prob = 1 - total_prob
            if rem_prob != 0:
                joint_table[i, 1:n] = 0
                joint_table = rem_prob/sum(joint_table)*joint_table
                joint_table[i, 1:n] = total_prob / n
                flag_vertical = 1
    
    if flag_vertical:
        results = []
        results.append(joint_table)
        results.append(hm_HypoTest)
        return results
    
    for i in range(n):
        index = np.argwhere(Y==i+1).T[0]
        p_val = getPvalue_Chisquare(X[index], Y[index], 1, m)
        if p_val > sig_level:
            total_prob = len(index)/len(Y)
            rem_prob = 1 - total_prob
            if rem_prob != 0:
                joint_table[1:m, i] = 0
                joint_table = rem_prob/sum(joint_table)*joint_table
                joint_table[1:m, i] = total_prob / m
    
    results = []
    results.append(joint_table)
    results.append(hm_HypoTest)
    return results
