# with input in the DataFrame format
# update metadata of the dataset

import os, sys
import typing
import scipy.io
import numpy as np
from sklearn import preprocessing
from common_primitives import utils
from d3m import container
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.metadata import params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
import rpi_d3m_primitives
from rpi_d3m_primitives.featSelect.Feature_Selector_model import JMI
from rpi_d3m_primitives.featSelect.RelationSet import RelationSet
from sklearn.impute import SimpleImputer


Inputs = container.DataFrame
Outputs = container.DataFrame

__all__ = ('JMIplus_auto',)

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    pass


class JMIplus_auto(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that performs supervised feature selection to reduce input feature dimension. Input to this primitive should be a matrix of tabular numerical/categorical data, consisting of columns of features, and an array of labels. Output will be a reduced data matrix with metadata updated.
    """
    
    metadata = metadata_base.PrimitiveMetadata({
        'id': '4d46e77b-9c29-4631-b382-02333e928f10',
        'version': '2.1.5',
        'name': 'JMIplus_auto feature selector',
        'keywords': ['Joint Mutual Information','Feature Selection'],
        'description': 'This algorithm is selecting the most relevant features based on the joint mutual inforamtion between features and the target.',
        'source': {
            'name': rpi_d3m_primitives.__author__,
            'contact': 'mailto:cuiz3@rpi.edu',
            'uris': [
                'https://github.com/zijun-rpi/d3m-primitives/blob/master/JMIplus_auto.py',
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
        'python_path': 'd3m.primitives.feature_selection.joint_mutual_information.AutoRPI',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MINIMUM_REDUNDANCY_FEATURE_SELECTION
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Union[typing.Dict[str, base.DockerContainer]] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._index = None
        self._problem_type = 'classification'
        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False
        self._cate_flag = None
        self._LEoutput = preprocessing.LabelEncoder()# label encoder
        self._Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # imputer
        self._nbins = 10
        self._Kbins = preprocessing.KBinsDiscretizer(n_bins=self._nbins, encode='ordinal', strategy='uniform') #KbinsDiscretizer
        
    ## TO DO:
    ## select columns via semantic types
    ## remove preprocessing
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        
        # set problem type
        metadata = outputs.metadata
        column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, 0))
        semantic_types = column_metadata.get('semantic_types', [])
        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            self._problem_type = 'classification'
            # set training labels
            self._LEoutput.fit(outputs)
            self._training_outputs = self._LEoutput.transform(outputs)
        else:
            self._problem_type = 'regression'
            
        # convert categorical values to numerical values in training data
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
                pass #replaces with zeros
            else:
                temp = list(inputs.iloc[:, column_index].values)
                for i in np.arange(len(temp)):
                    if bool(temp[i]):
                        self._training_inputs[i,column_index] = float(temp[i])
                    else:
                        ## TO DO: float nan
                        self._training_inputs[i,column_index] = float('nan')
                if not np.count_nonzero(np.isnan(self._training_inputs[:, column_index])) == 0:  # if there is missing values
                    if np.count_nonzero(np.isnan(self._training_inputs[:,column_index])) == m:   # all missing
                        self._training_inputs[:,column_index] = np.zeros(m,) # replace with all zeros
                        
        self._fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs.any() == None or self._training_outputs.any() == None: 
            raise ValueError('Missing training data, or missing values exist.')

        ## impute missing values
        self._Imputer.fit(self._training_inputs)
        self._training_inputs = self._Imputer.transform(self._training_inputs)
        
#        [m,n] = self._training_inputs.shape
#        for column_index in range(n):
#            if len(np.unique(self._training_inputs[:,column_index])) == 1:
#                self._cate_flag[column_index] = 1
                
        ## discretize non-categorical values
        disc_training_inputs = self._training_inputs
        if not len(np.where(self._cate_flag == 0)[0]) == 0:
            self._Kbins.fit(self._training_inputs[:, np.where(self._cate_flag == 0)[0]]) #find non-categorical values
            temp = self._Kbins.transform(self._training_inputs[:, np.where(self._cate_flag == 0)[0]])
            disc_training_inputs[:, np.where(self._cate_flag == 0)[0]] = temp
        #start from zero
        
        Trainset = RelationSet(self._training_inputs, self._training_outputs.reshape(-1, 1))
        discTrainset = RelationSet(disc_training_inputs, self._training_outputs.reshape(-1, 1))
        
        validSet, smallTrainSet = Trainset.split(self._training_inputs.shape[0] // 4)
        smallDiscTrainSet = discTrainset.split(self._training_inputs.shape[0] // 4)[1]
        model = JMI(smallTrainSet, smallDiscTrainSet,
                     self._problem_type, test_set=validSet)
        index = model.select_features()
        self._index = []
        [m, ] = index.shape
        for ii in np.arange(m):
            self._index.append(index[ii].item())
        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # inputs: m x n numpy array
        if self._fitted:
            output = inputs.iloc[:, self._index]
            output.metadata = utils.select_columns_metadata(inputs.metadata, columns=self._index)
            return CallResult(output)
        else:
            raise ValueError('Model should be fitted first.')


    def get_params(self) -> None:
        pass


    def set_params(self) -> None:
        pass

