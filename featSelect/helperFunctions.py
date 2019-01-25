import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

def normalize_array(vector, length):
    results = []
    min_value = 0
    num_states = 0

    if length == 0:
        length = vector.size
    normalized_vector = np.zeros(shape = (length,))

    if (length != 0) and (length != None):
        "Floor the vector elementwise and subtract the minimum elementwise"
        floored_vector = np.floor(np.array(vector))
        min_value = int(floored_vector.min())
        max_value = int(floored_vector.max())
        
        normalized_vector = floored_vector - min_value
        # Number of unique states
        num_states = max_value-min_value+1

    results.append(num_states)
    results.append(normalized_vector)
    return results


def joint(X):
	"""
		Takes in a matrix of values and maps each distinct row to a unique value,
			like "unordered inverse."

	"""
	if X.shape[1] == 0 or X.shape[1] == 1:
		M = X
	elif X.shape[0] == 0 or X.shape[1] == 1:
		M = X
	else:
		M = unordered_inverse(X)
	return M

def find_probs(observations):
	"""
	Takes a vector of nonnegative integral data points and returns a vector that 
		maps each data point to its probability ie find_probs(X)[i] will be P(X==i)
	"""
	if (observations.size == 0):
		return np.zeros(0)

	"Find the number of occurences for each data point"
	indices,counts = np.unique(observations,return_counts=True)
	indices = indices.astype(int)
	"Find the number of possible values, which will be the size of the result"
	max_value = indices[-1]
	num_states = int(max_value+1)
	"Find probabilities"
	probabilities = np.zeros(num_states)
	probabilities[indices] = counts
	probabilities /= observations.shape[0]
	return probabilities

def unordered_inverse(matrix):
	"""
		Takes in a matrix of values and maps each distinct row to a unique value
		ex 	[[1 2 3]	becomes [1,2,1] 
			 [4 5 6]
		 	 [1 2 3]]
	"""
	num_rows = matrix.shape[0]
	result = np.zeros((num_rows,))
	state_map = dict()
	state_count = 0
	for index,row in enumerate(matrix):
		row = row.tostring()
		if (row not in state_map):
			state_count += 1
			state_map[row] = state_count
		result[index] = state_map[row]
	return result

def counts(matrix):
	"""
		Takes in a matrix of values and returns the count of each unique row
		ex 	[[1 2 3]	becomes [2,1] (2 of [1 2 3], 1 of [4 5 6])
			 [4 5 6]
		 	 [1 2 3]]
	"""
	state_map = dict()
	for row in matrix:
		row = row.tostring()
		state_map[row] = state_map.get(row, 0)+1
	return np.array(list(state_map.values()))

def counts1d(vector):
	"""
		Takes in a vecore of values and returns the count of each unique value
		ex 	[1 2 3 1] becomes [2,1,1]
	"""
	states = np.arange(np.max(vector)+2)
	state_counts = np.histogram(vector,states)[0]
	state_counts = state_counts[state_counts.nonzero()]
	return state_counts

def get_score(train_data,train_labels,test_data,test_labels,problem_type):
	"""
		Returns the f1 score resulting from 3NN classification if problem_type = 'classification',
			or the mse from regression if problem_type = 'regression'
	"""
	if (problem_type=="classification"):
		predictor = KNeighborsClassifier(n_neighbors=3)
	else:
		predictor = KNeighborsRegressor(n_neighbors=3)
	predictor.fit(train_data,train_labels)
	predicted_labels = predictor.predict(test_data)

	if (problem_type=="regression"):
		score = mean_squared_error(test_labels,predicted_labels)
	else:
		score = accuracy_score(test_labels,predicted_labels)
	return score