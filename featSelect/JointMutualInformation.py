from rpi_d3m_primitives.featSelect.MergeArrays import mergeArrays
from rpi_d3m_primitives.featSelect.mutualInformation import mi
import numpy as np

def jmi(featureMatrix, classColumn, k):
    
    noOfFeatures = featureMatrix.shape[1]
    noOfSamples = featureMatrix.shape[0]
    
    classMI = np.zeros(shape = (noOfFeatures,))
    selectedFeatures = np.zeros(shape = (noOfFeatures,))
    
    sizeOfMatrix = k*noOfFeatures
    featureMIMatrix = np.zeros(shape = (int(sizeOfMatrix),))
    
    outputFeatures = np.zeros(shape = (int(k),))
    
    maxMI = 0
    maxMICounter = -1
    
    feature2D = [[]]*noOfFeatures
    
    for i in range(0, noOfFeatures):
        feature2D[i] = featureMatrix[:,i]
    
    for i in range(0, sizeOfMatrix):
        featureMIMatrix[i] = -1
        
    for i in range(0, noOfFeatures):
        classMI[i] = mi(feature2D[i], classColumn, noOfSamples)
        if classMI[i] > maxMI:
            maxMI = classMI[i]
            maxMICounter = i
    
    selectedFeatures[maxMICounter] = 1
    outputFeatures[0] = maxMICounter
    
    for i in range(1, k):
        score = 0
        currentHighestFeature = 0
        currentScore = 0
        #totalFeatureMI = 0
        
        for j in range(0, noOfFeatures):
            # if not select j
            if selectedFeatures[j] == 0:
                currentScore = 0
                #totalFeatureMI = 0                
                for x in range(0, i):
                    arrayPosition = x*noOfFeatures + j
                    
                    if featureMIMatrix[arrayPosition] == -1:
                        results = mergeArrays(feature2D[int(outputFeatures[x])], feature2D[j], noOfSamples)
                        mergedVector = results[1]
                        mutualinfo = mi(mergedVector, classColumn, noOfSamples)
                        featureMIMatrix[arrayPosition] = mutualinfo
                     
                    currentScore += featureMIMatrix[arrayPosition]
                     
                if currentScore > score:
                    score = currentScore
                    currentHighestFeature = j
                    
        selectedFeatures[currentHighestFeature] = 1
        outputFeatures[i] = currentHighestFeature
    
    outputFeatures = outputFeatures.astype(int)
    return outputFeatures
        