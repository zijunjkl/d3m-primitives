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
import time


Inputs = container.DataFrame
Outputs = container.DataFrame

__all__ = ('NaiveBayes_PointInf',)

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    pass


class NaiveBayes_PointInf(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which does naive bayes classification. MAP Point Inference is applied.
    """
    
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'e38a1b20-0262-4720-abec-9262a1fc0cf9',
        'version': '2.1.5',
        'name': 'Naive Bayes Classifier',
        'keywords': ['Naive Bayes','MAP Point Inference','Classification'],
        'description': 'This algorithm is an implementation of Naive Bayes classification with MAP Point Inference',
        'source': {
            'name': rpi_d3m_primitives.__author__,
            'contact': 'mailto:cuiz3@rpi.edu',
            'uris': [
                'https://github.com/zijun-rpi/d3m-primitives/blob/master/NaiveBayes_PointInf.py',
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
        'python_path': 'd3m.primitives.classification.naive_bayes.PointInfRPI',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.NAIVE_BAYES_CLASSIFIER],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, base.DockerContainer]] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._index = None
        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False
        self._target_columns_metadata: List[Dict] = None
        self._clf = Model('nb', bayesInf=1, PointInf=1)
        self._LEoutput = preprocessing.LabelEncoder()
    
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
        

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        
        # Update semantic types and prepare it for predicted targets
        self._store_target_columns_metadata(outputs)
        
        # set training labels
        metadata = outputs.metadata
        column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, 0))
        semantic_types = column_metadata.get('semantic_types', [])
        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            self._LEoutput.fit(outputs)
            self._training_outputs = self._LEoutput.transform(outputs)
        
        # convert categorical values to numerical values in training data
        metadata = inputs.metadata
        [m,n] = inputs.shape
        self._training_inputs = np.zeros((m,n))
        for column_index in metadata.get_elements((metadata_base.ALL_ELEMENTS,)):
            if column_index is metadata_base.ALL_ELEMENTS: 
                continue
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
                LE = preprocessing.LabelEncoder()
                LE = LE.fit(inputs.iloc[:,column_index])
                self._training_inputs[:,column_index] = LE.transform(inputs.iloc[:,column_index])  
            elif 'http://schema.org/Text' in semantic_types:
                pass
            else:
                temp = list(inputs.iloc[:, column_index].values)
                for i in np.arange(len(temp)):
                    if bool(temp[i]):
                        self._training_inputs[i,column_index] = float(temp[i])
                    else:
                        self._training_inputs[i,column_index] = 'nan'
        self._fitted = False
    

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs.any() == None or self._training_outputs.any() == None: 
            raise ValueError('Missing training data, or missing values exist.')

        discTrainset = RelationSet(self._training_inputs, self._training_outputs.reshape(-1, 1))
        discTrainset.impute()
        discTrainset.discretize() #starting from 1
        # if original lable value is starting from 1, same label after discretization, prediction + 1
        # if original label value is starting from 0, label += 1
        discTrainset.remove()
        X_train = discTrainset.data - 1
        Y_train = discTrainset.labels - 1 
        bins = discTrainset.NUM_STATES
        stateNo = np.append(bins, len(np.unique(Y_train)))
        t = time.time()
        self._clf.fit(X_train, Y_train, stateNo)
        self._time = time.time() - t
        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # inputs: m x n numpy array
        if self._fitted:
            # get discrete bins from training data
            discTrainset = RelationSet(self._training_inputs, self._training_outputs.reshape(-1, 1))
            discTrainset.impute()
            discTrainset.discretize()
            discTrainset.remove()
            bins = discTrainset.NUM_STATES
            
            # convert categorical values to numerical values in testing data
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
                    LE = LE.fit(inputs.iloc[:, column_index])
                    X_test[:, column_index] = LE.transform(inputs.iloc[:, column_index])
                elif 'http://schema.org/Text' in semantic_types:
                    pass
                else:
                    temp = list(inputs.iloc[:, column_index].values)
                    for i in np.arange(len(temp)):
                        if bool(temp[i]):
                            X_test[i, column_index] = float(temp[i])
                        else:
                            X_test[i, column_index] = 'nan'
            discTestset = RelationSet(X_test, [])
            discTestset.impute()
            X_test = discTestset.data
            index_list = np.setdiff1d(np.arange(discTrainset.num_features),np.array(discTrainset.removeIdx))
            X_test = X_test[:, index_list]
            est = preprocessing.KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy='uniform')
            est.fit(X_test)
            X_test = est.transform(X_test)
            t = time.time()
            output = self._clf.predict(X_test)
            self._time = time.time() - t + self._time
            if min(self._training_outputs) == 1:
                output = output + 1
            # label decode
            output = self._LEoutput.inverse_transform(output)
            
            # update metadata
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

