import os
import numpy as np
from d3m import container
from collections import OrderedDict
from d3m import container, utils
from common_primitives import utils as comUtils
from d3m.metadata import base as metadata_base
from d3m import metrics
from d3m.primitives.datasets import DatasetToDataFrame
from d3m.primitives.data import ExtractColumnsBySemanticTypes
from d3m.primitives.data import ColumnParser
from d3m.primitives.data import UnseenLabelEncoder, UnseenLabelDecoder
from common_primitives import construct_predictions
from common_primitives import compute_scores
import pandas as pd

from rpi_d3m_primitives.aSTMBplus import aSTMBplus
from rpi_d3m_primitives.aSTMBplus_auto import aSTMBplus_auto
from rpi_d3m_primitives.JMIplus import JMIplus
from rpi_d3m_primitives.JMIplus_auto import JMIplus_auto
from rpi_d3m_primitives.STMBplus import STMBplus
from rpi_d3m_primitives.STMBplus_auto import STMBplus_auto
from rpi_d3m_primitives.IPCMBplus import IPCMBplus
from rpi_d3m_primitives.IPCMBplus_auto import IPCMBplus_auto
from rpi_d3m_primitives.S2TMBplus import S2TMBplus
from rpi_d3m_primitives.NaiveBayes_PointInf import NaiveBayes_PointInf as NB_P
from rpi_d3m_primitives.NaiveBayes_BayesianInf import NaiveBayes_BayesianInf as NB_B
from rpi_d3m_primitives.TreeAugmentNB_PointInf import TreeAugmentNB_PointInf as TAN_P
from rpi_d3m_primitives.TreeAugmentNB_BayesianInf import TreeAugmentNB_BayesianInf as TAN_B
#import d3m.primitives.feature_selection.adaptive_simultaneous_markov_blanket as aSTMB
#import d3m.primitives.classification.naive_bayes as NB

dataset_name = '38_sick'
#dataset_name = '185_baseball'
#dataset_name = '27_wordLevels'

print('\ndataset to dataframe')   
# step 1: dataset to dataframe
path = os.path.join('/Users/zijun/Dropbox/', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))
hyperparams_class = DatasetToDataFrame.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFrame(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\nExtract Attributes')
# step 2: Extract Attributes
hyperparams_class = ExtractColumnsBySemanticTypes.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypes(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value

print('\nParse string into their types')
# step 3: Parsing 
hyperparams_class = ColumnParser.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ColumnParser(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=trainD)
trainD = call_metadata.value

# last column will be the true target for evaluation
target_idx = trainD.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']+1

print('\nExtract Targets')
# step 4: extract targets
hyperparams_class = ExtractColumnsBySemanticTypes.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypes(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

print('\nLabel Encoder')
# step 5: label encoder for target labels
encoder_hyperparams_class = UnseenLabelEncoder.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
encoder_primitive = UnseenLabelEncoder(hyperparams=encoder_hyperparams_class.defaults())
encoder_primitive.set_training_data(inputs=trainL)
encoder_primitive.fit()
trainL = encoder_primitive.produce(inputs=trainL).value


########################################################################################
print('\nFeature Selection')
#step 6 feature selection
hyperparams_class = STMBplus.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
FSmodel = STMBplus(hyperparams=hyperparams_class.defaults())
FSmodel.set_training_data(inputs=trainD, outputs=trainL)        
FSmodel.fit()
print('\nSelected Feature Index')
print(FSmodel._index)
print('\n')
trainD = FSmodel.produce(inputs=trainD) 
trainD = trainD.value


print ('Classification phase: Naive Bayes classifier')
hyperparams_class = NB_P.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
classifier = NB_P(hyperparams=hyperparams_class.defaults())
#print ('Classification phase: TAN classifier')
#hyperparams_class = TreeAugmentNB.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#classifier = TreeAugmentNB(hyperparams=hyperparams_class.defaults())

classifier.set_training_data(inputs=trainD, outputs=trainL)
classifier.fit()
########################################################################################
        

print ('\nLoad testing dataset') 
path = os.path.join('/Users/zijun/Dropbox/', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
#path = '/Users/zijun/Dropbox/38_sick/TEST/dataset_TEST/datasetDoc.json'
dataset = container.Dataset.load('file://{uri}'.format(uri=path))
hyperparams_class = DatasetToDataFrame.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFrame(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\nExtract Attributes')
hyperparams_class = ExtractColumnsBySemanticTypes.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypes(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value

print('\nParse string into their types')
hyperparams_class = ColumnParser.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ColumnParser(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=testD)
testD = call_metadata.value

print('\nSubset of testing data')
testD = FSmodel.produce(inputs=testD)
testD = testD.value

print('\nExtract Suggested Target')
hyperparams_class = ExtractColumnsBySemanticTypes.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypes(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

print('\nClassifier Prediction')
predictedTargets = classifier.produce(inputs=testD)
predictedTargets = predictedTargets.value
#predictedTargets.metadata = comUtils.select_columns_metadata(testL.metadata, columns=[0])

print('\nLabel Decoder')
decoder_hyperparams_class = UnseenLabelDecoder.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
decoder_primitive = UnseenLabelDecoder(hyperparams=decoder_hyperparams_class.defaults().replace({'encoder': encoder_primitive}))
predictedTargets = decoder_primitive.produce(inputs=predictedTargets).value

print('\nConstruct Predictions')
hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
dataframe = call_metadata.value

print('\ncompute scores')
path = os.path.join('/Users/zijun/Dropbox/', dataset_name, 'SCORE/dataset_TEST/datasetDoc.json')
#path = '/Users/zijun/Dropbox/38_sick/SCORE/dataset_TEST/datasetDoc.json'
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

#target_idx = dataset.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
dataset.metadata = dataset.metadata.add_semantic_type(('0', metadata_base.ALL_ELEMENTS, target_idx), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('0', metadata_base.ALL_ELEMENTS, target_idx), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
metrics_class = hyperparams_class.configuration['metrics'].elements
primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'F1_MACRO',
                'pos_label': None,
                'k': None,
            })],
        }))
scores = primitive.produce(inputs=dataframe, score_dataset=dataset).value

print(scores)


