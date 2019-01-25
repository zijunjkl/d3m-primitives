import numpy as np
from rpi_d3m_primitives.featSelect.helperFunctions import find_probs
from scipy.stats import entropy
from sklearn import preprocessing

def findOptimalSplitPoint(min_val, max_val, ori_feat, label, incre_rate = 0.1):    
    hm_bins = round(1/incre_rate)
    splits = np.linspace(min_val, max_val, hm_bins+1)
    hm_class = len(np.unique(label))
    if hm_class <= 1:
        min_entropy = 0
        optimal_split = max_val
                
        return optimal_split, min_entropy
    
    #edges = np.histogram(label, hm_class-1)[1]
    #label = np.digitize(label, edges)
    
    hm_sample = len(label)
    
    entropies = np.zeros(hm_bins,)
    
    for i in range(1, hm_bins+1):
        split = splits[i]
        left_indices = np.argwhere(ori_feat<split).flatten()
        left_labels = label[left_indices].flatten()
        left_probabilities = find_probs(left_labels)
        left_entropy = entropy(left_probabilities,base=2)

        right_indices = np.argwhere(ori_feat>=split).flatten()
        right_labels = label[right_indices].flatten()
        right_probabilities = find_probs(right_labels)
        right_entropy = entropy(right_probabilities,base=2)

        entropies[i-1] = left_labels.size/hm_sample * left_entropy + right_labels.size / hm_sample * right_entropy
    min_entropy = np.amin(entropies)
    idx = np.where(entropies == min_entropy)
    if len(idx[0]) != 1:
        idx = idx[0][0]
    else:
        idx = idx[0]          #only return idx for the first minimul element
    optimal_split = splits[idx+1]
        
    return optimal_split, min_entropy


def HillClimbing_entropy_discretization(feature, label, num_bins, relative_entropy_reduce_rate = 0.01):
    
    feature = feature.astype(np.float32)
    hm_class = np.unique(label).shape[0]
    min_val = np.min(feature)
    max_val = np.max(feature)
    min_label_val = int(np.min(label))
    
    curr_entropy = 0
    pre_entropy = 0
    
    "Calculate entropy of original distribution"
    probabilities = find_probs(label)
    pre_entropy = entropy(probabilities,base=2)
    

    if len(np.unique(feature)) > 15:
        init_splitset = np.linspace(min_val, max_val, num_bins+1)
        stop_flag = 0
        while stop_flag == 0:
            minentropy_inbin = np.zeros(num_bins,) #dim = (10,)
            curr_entropy = 0
            for s in range(1, num_bins):
                sub_min = init_splitset[s-1]
                sub_max = init_splitset[s+1]
                index = np.argwhere((sub_min<=feature) & (feature<sub_max)).flatten()
                if (len(index) != 0):
                    index = np.array(index)
                    feat = feature[index]
                    lab = label[index]
                else:
                    feat = []
                    lab = []
                init_splitset[s], minentropy_inbin[s-1] = findOptimalSplitPoint(sub_min, sub_max, feat, lab, 0.1)
            
            count_inbin = np.histogram(feature, init_splitset)[0]
            bins = np.digitize(feature, init_splitset)
            
            num_data = np.zeros(hm_class,)
            for n in range(num_bins):
                en = 0
                left_limit = init_splitset[n]
                right_limit = init_splitset[n+1]
                index = np.argwhere((left_limit<=feature) & (feature<right_limit)).flatten()
                select_labels = label[index].flatten()
                probabilities = find_probs(select_labels)
                ent = entropy(probabilities,base=2)
                curr_entropy = curr_entropy + ent*count_inbin[n]/feature.shape[0]
                
            if curr_entropy < 0.0000001:
                stop_flag = 1
                continue
            
            relative_reduction = (pre_entropy - curr_entropy) /pre_entropy
            if relative_reduction < relative_entropy_reduce_rate:
                stop_flag = 1
                
            pre_entropy = curr_entropy
        
        discretized_feature = bins
         
    else:
        hm_unique_state = len(np.unique(feature))
        init_splitset = hm_unique_state
        if hm_unique_state != 1:
            edges = np.histogram(feature, hm_unique_state-1)[1]
            discretized_feature = np.digitize(feature, edges)
        else:
            discretized_feature = feature
            
        curr_entropy = pre_entropy
        
    optimal_split_pointset = init_splitset
    final_entropy = curr_entropy
    return discretized_feature, optimal_split_pointset, final_entropy


def HC_discretization(trainD, trainL, hm_bins):
    samples,hm_features = trainD.shape

    #check the labels
    #hm_unique_class = len(np.unique(trainL))
    hm_unique_class = np.ceil(np.max(trainL)) - np.floor(np.min(trainL)) + 1
    edges = np.histogram(trainL, int(hm_unique_class) - 1)[1]
    #disc_trainL = np.digitize(trainL, edges)[:,0]  #dim = (samples,)
    disc_trainL = np.digitize(trainL, edges)
    disc_trainL = np.reshape(disc_trainL, (samples,1))
    
    #Discretize the features
    disc_trainD = np.zeros([samples, hm_features])

    optimal_split = []
    for i in range(hm_features):
        feature = trainD[:,i] #dim = (samples,)
        disc_feat,split,_ = HillClimbing_entropy_discretization(feature, disc_trainL, hm_bins, 0.01)
        optimal_split.append(split)
        le = preprocessing.LabelEncoder()
        le.fit(disc_feat)
        disc_trainD[:,i] = le.transform(disc_feat)+1 #dim = (samples,)
        #disc_trainD[:,i] = np.reshape(disc_feat, [samples,1])
                
    return disc_trainD, disc_trainL,optimal_split