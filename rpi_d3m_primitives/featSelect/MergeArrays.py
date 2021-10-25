from rpi_d3m_primitives.featSelect.helperFunctions import normalize_array
import numpy as np

def mergeArrays(firstVector, secondVector, length):
    
    if length == 0:
        length = firstVector.size
    
    results = normalize_array(firstVector, 0)
    #firstNumStates = results[0]
    #firstNormalisedVector = results[1]
    
    results = normalize_array(secondVector, 0)
    #secondNumStates = results[0]
    #secondNormalisedVector = results[1]
    
    stateCount = 1
    #stateMap = np.zeros(shape = (firstNumStates*secondNumStates,))
    merge = np.zeros(shape =(length,))

    joint_states = np.column_stack((firstVector,secondVector))
    uniques,merge = np.unique(joint_states,axis=0,return_inverse=True)
    stateCount = len(uniques)
    results = []
    results.append(stateCount)
    results.append(merge)
    return results

def mergeArrays2(firstVector, secondVector, length):
    
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
    
    for i in range(0, length):
        curIndex = firstNormalisedVector[i] + (secondNormalisedVector[i] * firstNumStates);
        if stateMap[int(curIndex)] == 0:
            stateMap[int(curIndex)] = stateCount
            stateCount = stateCount + 1
        merge[i] = stateMap[int(curIndex)]
    
    results = []
    results.append(stateCount)
    results.append(merge)
    
    return results