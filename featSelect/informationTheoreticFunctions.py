import numpy as np
from rpi_d3m_primitives.featSelect.helperFunctions import find_probs, normalize_array
from sklearn.metrics import mutual_info_score

def joint_probability(firstVector, secondVector, length):
    if length == 0:
         length = firstVector.size
    
    results = normaliseArray(firstVector, 0)
    firstNumStates = results[0]
    firstNormalisedVector = results[1]
    
    results = normaliseArray(secondVector, 0)
    secondNumStates = results[0]
    secondNormalisedVector = results[1]
    
    jointNumStates = firstNumStates * secondNumStates
    
    firstStateProbs = find_probs(firstNormalisedVector)
    secondStateProbs = find_probs(secondNormalisedVector)
    jointStateProbs = np.zeros(shape = (jointNumStates,))

    # Joint probabilities
    jointStates = np.column_stack((firstNormalisedVector,secondNormalisedVector))
    jointIndices,jointCounts = np.unique(jointStates,axis=0, return_counts = True)
    jointIndices = jointIndices.T
    jointIndices = jointIndices[1]*firstNumStates + jointIndices[0]
    jointIndices = jointIndices.astype(int)
    jointStateProbs[jointIndices] = jointCounts
    jointStateProbs /= length
    
    results = []
    results.append(jointStateProbs)
    results.append(jointNumStates)
    results.append(firstStateProbs)
    results.append(firstNumStates)
    results.append(secondStateProbs)
    results.append(secondNumStates)
    return results


def mi(dataVector, targetVector, length = 0):
    dataVector = dataVector.ravel()
    targetVector = targetVector.ravel()
    mi = mutual_info_score(dataVector,targetVector)/np.log(2)
    return mi