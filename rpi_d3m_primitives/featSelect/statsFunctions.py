import numpy as np
from scipy.stats import chi2
from rpi_d3m_primitives.featSelect.helperFunctions import counts1d

def getPvalue_Chisquare(X,Y,m,n):

    if not X.size or not Y.size:
        p_val = 1
        return p_val

    hm_samples = len(X)


    Cx,X = np.unique(X, return_inverse = True)
    Cy,Y = np.unique(Y, return_inverse = True)
    X = X+1
    Y = Y+1
    expt = hm_samples/(m*n)

    # joint_state_counts is a flattened joint distribution,
    # where joint_state_counts[x*n+y] = the number of occurences of (X=x,Y=y)
    m,n = len(Cx),len(Cy)
    joint_state_counts = np.zeros(m*n)
    joint_states = np.column_stack((X-1,Y-1))
    joint_indices,joint_counts = np.unique(joint_states,axis=0, return_counts=True)
    joint_indices = joint_indices.T[0]*n + joint_indices.T[1]
    joint_state_counts[joint_indices] = joint_counts

    # Get chi-square from joint_state_counts
    components = (joint_state_counts-expt)**2/expt
    chisquare = sum(components)
    p_val = 1-chi2.cdf(chisquare, m*n-1)
    return p_val

def get_num_value_HypoTest(X, Y, part_X, part_Y):
      hm_x = len(counts1d(X))
      hm_y = len(counts1d(Y))
      
      min_x = np.min(X)
      max_x = np.max(X)
      for i in range(len(part_X) - 1):
          all_right = min_x >= part_X[i] 
          all_left = max_x < part_X[i+1]
          all_left_inclusive = max_x <= part_X[i+1]
          if (all_right and all_left and (part_X[i+1] != max(part_X) or (part_X[i+1] == max(part_X) and part_X[-1] == part_X[-2]))):
              hm_x = part_X[i+1] - part_X[i]
          elif all_right and all_left_inclusive and part_X[-1] != part_X[-2] and part_X[i+1] == max(part_X):
              hm_x = part_X[i+1] - part_X[i] + 1
          else:
              continue

      min_y = np.min(Y)
      max_y = np.max(Y)
      for i in range(len(part_Y) - 1):
          all_right = min_y >= part_Y[i] 
          all_left = max_y < part_Y[i+1]
          all_left_inclusive = min_y <= part_Y[i+1]
          if all_right and all_left and (part_Y[i+1] != max(part_Y) or (part_Y[i+1] == max(part_Y) and part_Y[-1] == part_Y[-2])):
              hm_y = part_Y[i+1] - part_Y[i]
          
          elif all_right and all_left_inclusive and part_Y[-1] != part_Y[-2] and part_Y[i+1] == max(part_Y):
              hm_y = part_Y[i+1] - part_Y[i] + 1
          else:
              continue

      return hm_x,hm_y


def getSquaredError(vector, edges):
    uni_val = np.unique(vector) #uni_val is a vector
    hm_val = len(uni_val)
    
    min_err = 0
    
    #if there is only one state
    if hm_val == 1:
        min_err_left = 0
        min_err_right = np.inf
        best_split = uni_val[0] + 1
        for e in range(len(edges)-1):
            if best_split > edges[e] and best_split <= edges[e+1]:
                if(best_split - edges[e] != 0):
                    expect = len(vector)/(best_split - edges[e])
                    min_err_left = (len(vector) - expect) ** 2

                
        best_split = uni_val[0]
        for e in range(len(edges)-1):
            if best_split > edges[e] and best_split <= edges[e+1]:
                if(best_split - edges[e+1] != 0):
                    expect = len(vector) / (edges[e+1] - best_split)
                    min_err_right = (len(vector) - expect) ** 2
            
        
        if min_err_left <= min_err_right:
            min_err = min_err_left
            best_split = uni_val[0] + 1
            results = []
            results.append(best_split)
            results.append(min_err)
            return results
        
        else:
            min_err = min_err_right
            best_split = uni_val[0]
            results = []
            results.append(best_split)
            results.append(min_err)
            return results
        
    #if binary case   
    isbinary = (hm_val == 2)
    
    uni_val = sorted(uni_val)
    SEs = []
    for i in range(hm_val):
        s = uni_val[i]
        if s == 1:
            if not (edges[1] - 1) == 0:
                expect = len(vector)/(edges[1] - 1) 
            else:
                expect = 0
            min_err = (len(vector) - expect) ** 2
            SEs.append(min_err)
            continue
        temp = [1,s,max(uni_val)+1]
        temp.sort()
        temp = np.asarray(temp)
        N = np.histogram(vector, temp)[0]
        expt_right = 0
        expt_left = 0
        for e in range(len(edges)):
            if s > edges[e] and s <= edges[e+1]:
                expt_left = N[0]/(s-edges[e])
                if e + 1 > len(edges):
                    expt_right = N[1]
                else:
                    if (edges[e+1] - s != 0):
                        expt_right = N[1] /(edges[e+1] - s)
                break
            
        if isbinary:
            expt_temp = sum(N)/2
            expt_left = expt_temp
            expt_right = expt_temp
            
        
        error_left = 0
        error_right = 0
        indices,counts = np.unique(vector.flatten(),return_counts=True)
        split_index = np.searchsorted(indices,s)
        # Computing squared error for indices left of the split
        left_counts = counts[:split_index]
        left_errors = (left_counts-expt_left)**2
        error_left = sum(left_errors)
        # Computing squared error for indices right of the split
        right_counts = counts[split_index:]
        right_errors = (right_counts-expt_right)**2
        error_right = sum(right_errors)  

        SEs.append(error_left + error_right)
        
    idx = np.argsort(SEs)
    SEs = sorted(SEs)
        
    for i in range(len(SEs)):
        best_split = uni_val[idx[i]]
        flag = sum(best_split == edges)
        if (best_split < max(edges) and flag == 0) or (best_split == max(edges) and flag == 1):
            min_err = SEs[i]
            results = []
            results.append(best_split)
            results.append(min_err)
            return results