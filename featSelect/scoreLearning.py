import numpy as np
from scipy.special import gammaln
from rpi_d3m_primitives.featSelect.helperFunctions import counts1d, counts, joint

def find_optimal_BN(context,training_set,score_cache):
	num_samples,num_feats = training_set.shape
	MAX_PARENTS = 5
	if (num_feats <= 2):
		MAX_PARENTS = 2
	adj_matrix = np.array([[0]*num_feats]*num_feats)
	for child in range(1,num_feats):
		parents = np.array([0]*num_feats)
		parent_value_old = -np.inf
		parent_count = 0
		while (parent_count < MAX_PARENTS):
			local_max = -np.inf
			for parent_candidate in range(child-1,-1,-1):
				if (parents[parent_candidate] == 0):
					# Candidate is not yet a parent of 'child'
					parents[parent_candidate] = 1
					parent_indices = np.argwhere(parents==1)
					local_score,score_cache = k2(child,parent_indices,context,training_set,score_cache)
					if local_score > local_max:
						local_max = local_score
						best_parent = parent_candidate
					parents[parent_candidate] = 0
			parent_value_new = local_max
			if (parent_value_old >= parent_value_new):
				break
			parents[best_parent] = 1
			parent_value_old = parent_value_new
			parent_count += 1
		adj_matrix[:,child] = parents
	return adj_matrix




def k2(child,parents,context,data,cache):
	parents = parents.flatten()
	true_child = context[child]
	true_parents = context[parents].tostring()
	joined = (true_child,true_parents)
	if (joined in cache.joint_cache):
		return cache.joint_cache[joined],cache

	child_cache = cache.child_cache
	child_states = data[:,child]
	child_states = child_states.astype(int)
	if (true_child in child_cache):
		child_state_count = child_cache[true_child]
	else:
		child_state_count = counts1d(child_states).size
		child_cache[true_child] = child_state_count

	parent_cache = cache.parent_cache
	if (true_parents in parent_cache):
		parent_states,parent_state_counts = parent_cache[true_parents]
	else:
		parent_states = data[:,parents]
		parent_states = parent_states.astype(int)
		parent_states = joint(parent_states)
		parent_state_counts = counts1d(parent_states)
		parent_cache[true_parents] = (parent_states,parent_state_counts)

	joint_states = np.column_stack((child_states,parent_states))
	joint_state_counts = counts(joint_states)

	k2 = len(parent_state_counts)*gammaln(child_state_count)
	for p_count in parent_state_counts:
		k2 -= gammaln(p_count+child_state_count)
	for j_count in joint_state_counts:
		if (j_count>1):
			k2 += gammaln(j_count+1)
	cache.joint_cache[joined] = k2
	return k2,cache




def k2_slow(data, child, parents):
	if (parents.size == 0):
		return 0
	child_states = data[:,child]
	child_states = child_states.astype(int)
	child_state_count = counts1d(child_states).size

	parent_states = data[:,parents]
	parent_states = parent_states.astype(int)
	parent_states = joint(parent_states)
	parent_state_counts = counts1d(parent_states)

	joint_states = np.column_stack((child_states,parent_states))
	joint_state_counts = counts(joint_states)

	k2 = len(parent_state_counts)*gammaln(child_state_count)
	for p_count in parent_state_counts:
		k2 -= gammaln(p_count+child_state_count)
	for j_count in joint_state_counts:
		if (j_count>1):
			k2 += gammaln(j_count+1)
	return k2