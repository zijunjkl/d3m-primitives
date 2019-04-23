import numpy as np
from rpi_d3m_primitives.featSelect.helperFunctions import normalize_array, joint
from rpi_d3m_primitives.featSelect.mutualInformation import mi, joint_probability
"""---------------------------- CONDITIONAL MUTUAL INFORMATION ----------------------------"""
def mergeArrays(firstVector, secondVector, length):
    
    if length == 0:
        length = firstVector.size
    
    results = normalize_array(firstVector, 0)
    firstNumStates = results[0]
    firstNormalisedVector = results[1]
    
    results = normalize_array(secondVector, 0)
    secondNumStates = results[0]
    secondNormalisedVector = results[1]
    
    stateCount = 1
    stateMap = np.zeros(shape = (firstNumStates*secondNumStates,))
    merge = np.zeros(shape =(length,))

    joint_states = np.column_stack((firstVector,secondVector))
    uniques,merge = np.unique(joint_states,axis=0,return_inverse=True)
    stateCount = len(uniques)
    results = []
    results.append(stateCount)
    results.append(merge)
    return results


def conditional_entropy(dataVector, conditionVector, length):
    condEntropy = 0
    jointValue = 0
    condValue = 0
    if length == 0:
        length = dataVector.size
    
    results = joint_probability(dataVector, conditionVector, 0)
    jointProbabilityVector = results[0]
    numJointStates = results[1]
    numFirstStates = results[3]
    secondProbabilityVector = results[4]
    
    for i in range(0, numJointStates):
        jointValue = jointProbabilityVector[i]
        condValue = secondProbabilityVector[int(i / numFirstStates)]
        if jointValue > 0 and condValue > 0:
            condEntropy -= jointValue * np.log2(jointValue / condValue);

    return condEntropy


def cmi(dataVector, targetVector, conditionVector, length = 0):
    if (conditionVector.size == 0):
        return mi(dataVector,targetVector,0)
    if (len(conditionVector.shape)>1 and conditionVector.shape[1]>1):
        conditionVector = joint(conditionVector)
    cmi = 0;
    firstCondition = 0
    secondCondition = 0
    
    if length == 0:
        length = dataVector.size
    
    results = mergeArrays(targetVector, conditionVector, length)
    mergedVector = results[1]
    
    firstCondition = conditional_entropy(dataVector, conditionVector, length)
    secondCondition = conditional_entropy(dataVector, mergedVector, length)
    cmi = firstCondition - secondCondition
    
    return cmi