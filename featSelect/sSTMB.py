import numpy as np
from rpi_d3m_primitives.featSelect.mutualInformation import mi
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.pyBN.learning.structure.score.hill_climbing import hc
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from rpi_d3m_primitives.featSelect.scoreLearning import find_optimal_BN
from rpi_d3m_primitives.featSelect.ScoreCache import ScoreCache
from rpi_d3m_primitives.featSelect.helperFunctions import joint

"""---------------------------- DRIVER FUNCTIONS FOR sSTMB ----------------------------"""
def sSTMBplus(train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,problem_type):
  best_score = 0
  best_features = np.array([])
  for fold in range(3):
    feats,score = analyze_sSTMB(fold,train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,problem_type)
    if (score>best_score):
      best_features = feats
      best_score = score
  return best_features

def sSTMB(train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,problem_type):
  NUM_PERMUTATIONS = 2
  PERMUTATION_SIZE = 16
  num_feats = train_data.shape[1]
  if (num_feats<20):
    return tian_sSTMB_new(Dtrain_data,Dtrain_labels)
  "We have to permute subsets of the features"
  selected_feats = [np.array([])]*NUM_PERMUTATIONS
  for i in range(NUM_PERMUTATIONS):
    permutation = np.random.permutation(num_feats)[:PERMUTATION_SIZE]
    feature_indices = tian_sSTMB_new(Dtrain_data[:,permutation],Dtrain_labels).astype(int)
    selected_feats[i] = permutation[feature_indices]
  best_accuracy = 0
  best_features = np.array([]).astype(int)
  for i in range(1,2**NUM_PERMUTATIONS):
    combination = [int(x) for x in list('{0:0b}'.format(i).zfill(NUM_PERMUTATIONS))]
    features = np.array([])
    for index,val in enumerate(combination):
      if (val):
        features = np.union1d(features,selected_feats[index])
    features = features.astype(int)
    if (features.size != 0):
      selected_train_data = train_data[:,features]
      selected_test_data = test_data[:,features]
      accuracy = get_score(selected_train_data,train_labels,selected_test_data,test_labels,problem_type)
    else:
      accuracy = 0
    if accuracy>best_accuracy:
      best_accuracy = accuracy
      best_features = features
  return best_features

"""---------------------------- sSTMB ALGORITHM ----------------------------"""
def tian_sSTMB_new(data, targets):
  SCORING_METRIC = 'K2'
  score_cache = ScoreCache()
  num_feats = data.shape[1]
  max_size = 1
  complete_data = np.concatenate((targets,data),axis=1)
  current_PC = RecognizePC_score(data,targets,score_cache,SCORING_METRIC)
  candidate_MBs = np.arange(1,num_feats)
  candidate_spouses = np.setdiff1d(candidate_MBs,current_PC)
  spouse_set = []
  target_index = np.array([0])
  while (candidate_spouses.size != 0):
    v = candidate_spouses[0]
    candidate_spouses = np.setdiff1d(candidate_spouses,v)
    Z = np.union1d(current_PC,np.union1d(v, target_index))
    new_size = Z.shape[0]
    maxsize = max(new_size,max_size)
    _,_,current_PC,spouses = local_learn_MB(Z,complete_data,score_cache,SCORING_METRIC)
    spouses = np.setdiff1d(spouses,current_PC)
    spouse_set = np.hstack((spouse_set,spouses))
  spouse_set = np.array(spouse_set).astype(int)
  if (spouse_set.size != 0):
    current_spouses = sort_by_cmi(spouse_set,targets,current_PC,complete_data)
  else:
    current_spouses = spouse_set

  true_spouses = np.array([])
  saveS = [[]]*num_feats
  Z = np.union1d(current_PC, 0)
  while (current_spouses.size != 0):
    v = current_spouses[0]
    current_spouses = np.setdiff1d(current_spouses,v)
    Z = np.union1d(np.union1d(current_PC,v),np.union1d(target_index,true_spouses))
    Z = Z.astype(int)
    new_size = true_spouses.shape[0]
    maxsize = max(new_size,max_size)
    _,children,current_PC,spouses = local_learn_MB(Z,complete_data,score_cache,SCORING_METRIC)
    # for child in children:
    #   if (current_spouses.size == 0):
    #     saveS[child] = list(np.setdiff1d(spouses,current_PC))
    true_spouses = np.setdiff1d(spouses,current_PC)
  MB = np.union1d(true_spouses,current_PC)-1
  return MB

def RecognizePC_score(data,targets,score_cache,SCORING_METRIC):
    num_feats = data.shape[1]
    max_size = 0
    sep_set = [[]]*data.shape[1]
    current_PC = np.array([]) # known as 'H' in MATLAB
    complete_data = np.concatenate((targets,data),axis=1)
    complete_data = complete_data.astype(int)
    candidates = np.arange(1,num_feats+1)
    Z = np.array([])
    target_index = np.array([0])
    while (candidates.size != 0):
        new_size = current_PC.shape[0]
        max_size = max(new_size,max_size)
        if (max_size>20):
            continue
        candidates = sort_by_cmi(candidates,targets,current_PC,complete_data)
        current_feat = candidates[0]
        candidates = np.setdiff1d(candidates,current_feat)
        Z = np.union1d(target_index,np.union1d(current_PC,current_feat))
        _,_,current_PC,_ = local_learn_PC(Z,complete_data,score_cache,SCORING_METRIC)
        removed_V = np.setdiff1d(np.setdiff1d(Z,current_PC),target_index)
        if current_feat not in current_PC:
            sep_set[current_feat-1] = np.union1d(sep_set[current_feat-1],current_PC)
        for i in range(len(removed_V)):
            sep_set[i] = np.union1d(sep_set[i],current_PC)
    Z = np.union1d(current_PC,target_index)
    _,_,final_PC,_ = local_learn_PC(Z,complete_data,score_cache,SCORING_METRIC)
    return final_PC

"""---------------------------- sSTMB HELPER FUNCTIONS ----------------------------"""
def local_learn_PC(Z,data,score_cache,metric="K2"):
  Z = Z.astype(int)
  if (metric in ["AIC","BIC","LL"]):
    BN = hc(data[:,Z],metric = metric)
    parents = BN.parents(0)
    children = BN.children(0)
  elif (metric in ["BD","K2"]):
    BN = find_optimal_BN(Z,data[:,Z],score_cache)
    parents = np.arange(len(Z))[BN[:,0].astype(bool)]
    children = np.arange(len(Z))[BN[0,:].astype(bool)]
  PC = np.union1d(Z[children],Z[parents])
  return parents,children,PC,BN

def local_learn_MB(Z,data,score_cache,metric="K2"):
  Z = Z.astype(int)
  parents,children,PC,BN = local_learn_PC(Z,data,score_cache,metric)
  spouses = np.array([])
  for child in children:
    if (metric in ["AIC","BIC","LL"]):
        spouses = np.union1d(BN.parents(child),spouses)
    elif (metric in ["BD","K2"]):
        new_spouses = np.arange(len(Z))[BN[:,child].astype(bool)]
        spouses = np.union1d(spouses,new_spouses)
  spouses = index(Z,np.setdiff1d(spouses,0))
  return parents,children,PC,spouses

def analyze_sSTMB(fold,train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,problem_type):
  train_samples = train_data.shape[0]
  lower = (fold)*train_samples//3
  upper = (fold+1)*train_samples//3
  train_data = train_data[lower:upper,:]
  train_labels = train_labels[lower:upper]
  Dtrain_data = Dtrain_data[lower:upper,:]
  Dtrain_labels = Dtrain_labels[lower:upper]
  test_samples = test_data.shape[0]
  lower = (fold)*test_samples//3
  upper = (fold+1)*test_samples//3
  test_data = test_data[lower:upper,:]
  test_labels = test_labels[lower:upper]
  selected_feats = sSTMB(train_data,train_labels,test_data,test_labels,Dtrain_data,Dtrain_labels,problem_type)
  if (selected_feats.shape[0] == 0):
    return selected_feats,0
  selected_feats = selected_feats.astype(int)
  selected_train_data = train_data[:,selected_feats]
  selected_test_data = test_data[:,selected_feats]
  score = get_score(selected_train_data,train_labels,selected_test_data,test_labels,problem_type)
  return selected_feats,score

def get_score(train_data,train_labels,test_data,test_labels, problem_type):
  if (problem_type=="classification"):
    predictor = KNeighborsClassifier(n_neighbors=5)
  else:
    predictor = KNeighborsRegressor(n_neighbors=5)
  predictor.fit(train_data,train_labels[:,0])
  predicted_labels = predictor.predict(test_data)

  if (problem_type=="classification"):
    score = accuracy_score(test_labels,predicted_labels)
  else:
    score = 1/(mean_squared_error(test_labels,predicted_labels) + 1e-10)
  return score



"""---------------------------- sSTMB ROUTINES----------------------------"""
def sort_by_cmi(feat_indices,targets,cond_indices,data):
  """
    Returns the indices found in 'feat_indices' in order of I(X;Y|Z), where Z 
      is the joint distribution described by data[cond_indices], X is the joint
      distribution of data[feat_indices[i]], and Y is the joint distribution of 
      data[targets]. If Z is empty, then the result is I(X;Y) 
  """
  feats_to_cmi = dict()
  if (cond_indices.size == 0):
    for feature in feat_indices:
      feats_to_cmi[feature] = mi(data[:,feature],targets)
  else:
    for feature in feat_indices:
      feats_to_cmi[feature] = cmi(data[:,feature],targets,joint(data[:,cond_indices]))

  sorted_features = np.array(sorted(feat_indices, key = lambda f: -feats_to_cmi[f]))
  return sorted_features

def index(array,indices):
  """
    Allows indexing with an empty array, which returns an empty array
  """
  indices = indices.astype(int)
  if (indices.size == 0):
    return np.array([])
  return array[indices]