# with input in the DataFrame format
# update metadata of the dataset
import os, sys
import typing
import scipy.io
import numpy as np
from sklearn import preprocessing
from collections import OrderedDict
from typing import cast, Any, Dict, List, Union, Sequence, Optional, Tuple
#from common_primitives import utils
from d3m import container
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.metadata import params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from rpi_d3m_primitives.structuredClassifier.structured_Classify_model import Model
from rpi_d3m_primitives.featSelect.RelationSet import RelationSet
import rpi_d3m_primitives
#import time
from sklearn.impute import SimpleImputer


Inputs = container.DataFrame
Outputs = container.DataFrame

__all__ = ('TreeAugmentedNB_BayesianInf',)

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    nbins = hyperparams.UniformInt(
            lower=2,
            upper=21,
            default=10,
            description = 'The number of bins for discretization.',
            semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )
    N0 = hyperparams.UniformInt(
        lower = 1,
        upper = 11,
        default = 5,
        description='The hyperparameter specifing the power of the prior. The larger the N0, the stronger the power of prior',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )


class TreeAugmentedNB_BayesianInf(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
     A primitive which does naive bayes classification. During training, input to this primitive should be a matrix of tabular numercal/categorical data, and an array of labels. During testing, input is data matrix of numerical features only and output will be the predicted labels with metadata generated.
    """
    
    metadata = metadata_base.PrimitiveMetadata({
        'id': '2fa0afb2-1b7b-462d-a7c9-11b44efe9eb0',
        'version': rpi_d3m_primitives.__coreversion__,
        'name': 'Tree-Augmented Naive Bayes Classifier',
        'keywords': ['Tree-Augmented Naive Bayes','Bayesian Inference','Classification'],
        'description': 'This algorithm is an implementation of Tree-augmented Naive Bayes classification. Bayesian Inference is applied.',
        'source': {
            'name': rpi_d3m_primitives.__author__,
            'contact': 'mailto:cuiz3@rpi.edu',
            'uris': [
                'https://github.com/zijun-rpi/d3m-primitives/blob/master/TreeAugmentNB_BayesianInf.py',
                'https://github.com/zijun-rpi/d3m-primitives.git'
                ]
        },
        'installation':[
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'rpi_d3m_primitives',
	            'version': rpi_d3m_primitives.__version__
            }
        ],
        'python_path': 'd3m.primitives.classification.tree_augmented_naive_bayes.BayesianInfRPI',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.NAIVE_BAYES_CLASSIFIER],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION
    })
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, base.DockerContainer]] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._index = None
        self._training_inputs = None
        self._training_outputs = None
        self._origin_inputs = None #for label encoder
        self._fitted = False
        self._cate_flag = None
        self._clf = Model(modelName='tan', bayesInf=1, PointInf=1, alpha=1, N0=self.hyperparams['N0']) #classifier
        self._LEoutput = preprocessing.LabelEncoder() #label encoder
        self._Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #imputer
        self._nbins = self.hyperparams['nbins']
        self._Kbins = preprocessing.KBinsDiscretizer(n_bins=self._nbins, encode='ordinal', strategy='uniform') #KbinsDiscretizer
        self._discTrainset = None
        
    
    def _store_target_columns_metadata(self, outputs: Outputs) -> None:
        outputs_length = outputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[Dict] = []

        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs.metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = list(column_metadata.get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            semantic_types = [semantic_type for semantic_type in semantic_types if semantic_type != 'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)
            
        self._target_columns_metadata = target_columns_metadata
        
    ##TO DO:
    #select columns via semantic types
    #remove preprocessing
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:

        ## Update semantic types and prepare it for predicted targets
        self._store_target_columns_metadata(outputs)
        
        ## memory original training inputs
        self._origin_inputs = inputs

        ## set training labels
        metadata = outputs.metadata
        column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, 0))
        semantic_types = column_metadata.get('semantic_types', [])
        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            self._LEoutput.fit(outputs)
            self._training_outputs = self._LEoutput.transform(outputs) #starting from zero
        
        ## convert categorical values to numerical values in training data
        metadata = inputs.metadata
        [m,n] = inputs.shape
        self._training_inputs = np.zeros((m,n))
        self._cate_flag = np.zeros((n,))
        for column_index in metadata.get_elements((metadata_base.ALL_ELEMENTS,)):
            if column_index is metadata_base.ALL_ELEMENTS: 
                continue
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
                LE = preprocessing.LabelEncoder()
                LE = LE.fit(inputs.iloc[:,column_index])
                self._training_inputs[:,column_index] = LE.transform(inputs.iloc[:,column_index])
                self._cate_flag[column_index] = 1
            elif 'http://schema.org/Text' in semantic_types:
                pass
            else:
                temp = list(inputs.iloc[:, column_index].values)
                for i in np.arange(len(temp)):
                    if bool(temp[i]):
                        self._training_inputs[i,column_index] = float(temp[i])
                    else:
                        self._training_inputs[i,column_index] = float('nan')
                # imputer will remove the column with purely missing values
                if not np.count_nonzero(np.isnan(self._training_inputs[:, column_index])) == 0:  # if there is missing values
                    if np.count_nonzero(np.isnan(self._training_inputs[:, column_index])) == m:  # all missing
                        self._training_inputs[:, column_index] = np.zeros(m, )  # replace with all zeros

        self._fitted = False
    

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs.any() == None or self._training_outputs.any() == None: 
            raise ValueError('Missing training data, or missing values exist.')

        ## impute missing values
        self._Imputer.fit(self._training_inputs)
        self._training_inputs = self._Imputer.transform(self._training_inputs)

        ## discretize non-categorical values
        disc_training_inputs = self._training_inputs
        if not len(np.where(self._cate_flag == 0)[0]) == 0:
            self._Kbins.fit(self._training_inputs[:, np.where(self._cate_flag == 0)[0]]) #find non-categorical values
            temp = self._Kbins.transform(self._training_inputs[:, np.where(self._cate_flag == 0)[0]])
            disc_training_inputs[:, np.where(self._cate_flag == 0)[0]] = temp
        # starting from zero

        ## get number of states for each feature and remove features with only one state
        discTrainset = RelationSet(disc_training_inputs, self._training_outputs.reshape(-1,1))
        discTrainset.getStateNo(self._cate_flag, self._nbins)
        discTrainset.remove()
        X_train = discTrainset.data
        Y_train = discTrainset.labels
        
        self._discTrainset = discTrainset
        stateNo = np.append(discTrainset.NUM_STATES, len(np.unique(Y_train)))

        ## fit the classifier
        self._clf.fit(X_train, Y_train, stateNo)
        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # inputs: m x n numpy array
        if self._fitted:

            ## convert categorical values to numerical values in testing data
            metadata = inputs.metadata
            [m, n] = inputs.shape
            X_test = np.zeros((m, n))
            for column_index in metadata.get_elements((metadata_base.ALL_ELEMENTS,)):
                if column_index is metadata_base.ALL_ELEMENTS:
                    continue
                column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
                semantic_types = column_metadata.get('semantic_types', [])
                if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
                    LE = preprocessing.LabelEncoder()
                    LE = LE.fit(self._origin_inputs.iloc[:, column_index]) #use training data to fit
                    X_test[:, column_index] = LE.transform(inputs.iloc[:, column_index])
                elif 'http://schema.org/Text' in semantic_types:
                    pass
                else:
                    temp = list(inputs.iloc[:, column_index].values)
                    for i in np.arange(len(temp)):
                        if bool(temp[i]):
                            X_test[i, column_index] = float(temp[i])
                        else:
                            X_test[i, column_index] = float('nan')

            ## impute testing data
            X_test = self._Imputer.transform(X_test)

            ## Kbins discretize for noncategorical values
            disc_X_test = X_test
            if not len(np.where(self._cate_flag == 0)[0]) == 0:
                temp = self._Kbins.transform(X_test[:, np.where(self._cate_flag == 0)[0]])
                disc_X_test[:,np.where(self._cate_flag == 0)[0]] = temp

            ## remove columns with one states
            index_list = np.setdiff1d(np.arange(self._discTrainset.num_features), np.array(self._discTrainset.removeIdx))
            disc_X_test = disc_X_test[:, index_list]

            ## prediction
            output = self._clf.predict(disc_X_test)

            ## label decode
            output = self._LEoutput.inverse_transform(output)
            
            ## update metadata
            output = container.DataFrame(output, generate_metadata=False, source=self)
            output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)
            
            for column_index, column_metadata in enumerate(self._target_columns_metadata):
                output.metadata = output.metadata.update_column(column_index, column_metadata, source=self)


            return CallResult(output)
        else:
            raise ValueError('Model should be fitted first.')


    def get_params(self) -> None:
        pass


    def set_params(self) -> None:
        pass

