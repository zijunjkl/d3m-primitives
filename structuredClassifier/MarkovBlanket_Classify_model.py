import sys
import os
from rpi_d3m_primitives.structuredClassifier.learnCPD import learnCPDAllD
from rpi_d3m_primitives.structuredClassifier.inference import posteriorInf
from rpi_d3m_primitives.pyBN.learning.structure.naive.naive_bayes import naive_bayes
from rpi_d3m_primitives.pyBN.learning.structure.naive.TAN import TAN
#from rpi_d3m_primitives.pyBN.learning.structure.score.hill_climbing import hc as hill_climbing
#from rpi_d3m_primitives.pyBN.learning.structure.score import tabu
#from rpi_d3m_primitives.pyBN.learning.structure.constraint.grow_shrink import gs as grow_shrink
#from rpi_d3m_primitives.pyBN.learning.structure.hybrid.mmhc import mmhc
import numpy as np

class Model():
    
    def __init__( self, bayesInf, PointInf, alpha, N0):
        
#        self.modelName = modelName
        self.bayesInf = bayesInf
        self.PointInf = PointInf
        self.N0 = N0
        self.alpha = alpha
        self.stateNo = []
        self.parents = []
        self.CPD = []
        self.score = []

    
    def learnStructure( self, train_data, train_labels, **kwargs):
        
        trainMatrix = np.concatenate( [train_data, train_labels.reshape(-1,1)], 1)
        D = trainMatrix.shape[1]
        if self.modelName == 'nb':
            bn = naive_bayes(trainMatrix, D-1)
        elif self.modelName == 'tan':
            bn = TAN(trainMatrix, D-1)
        #elif self.modelName == 'hc':
        #    bn = hill_climbing(trainMatrix)
        #elif self.modelName == 'Tabu':
        #    bn = tabu(trainMatrix)
        #elif self.modelName == 'gs':
        #    bn = grow_shrink(trainMatrix)
        #elif self.modelName == 'mmhc':
        #    bn = mmhc(trainMatrix)

        for i in range(D):
            self.parents.append(bn.parents(i))
        
    def learnParameters( self, train_data, train_labels, bayesInf, PointInf, debug= False):
        
        self.bayesInf = bayesInf
        self.PointInf = PointInf
        if len(self.parents) == 0:
            print('Error')
        else:    
            trainMatrix = np.concatenate( [train_data, train_labels.reshape(-1,1)], 1)
            D = trainMatrix.shape[1]
            
            if debug == True:
                for i in range( D):            
                    if len( np.unique( trainMatrix[:,i])) != self.stateNo[i]:
                        print('Error ' + str(len( np.unique( trainMatrix[:,i]))) + ' != ' +str(self.stateNo[i]))
                        
            self.CPD = learnCPDAllD( trainMatrix, self.stateNo, self.parents, alpha = self.alpha, N0 = self.N0, bayesInf = self.bayesInf, PointInf = self.PointInf)    
        
    def fit( self, train_data, train_labels, stateNo, debug= False, **kwargs):
        self.stateNo = stateNo
#        self.learnStructure( train_data, train_labels, **kwargs)   
        self.learnParameters( train_data, train_labels, self.bayesInf, self.PointInf)        
        
    def predict( self, test_data):
        
        D = len( self.CPD)
        condP, Yest = posteriorInf( test_data, self.parents, self.CPD, D-1)
        
        return Yest