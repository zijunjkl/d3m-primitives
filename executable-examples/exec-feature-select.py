import os
import numpy as np
from d3m import container
from collections import OrderedDict
from d3m import container, utils
from common_primitives import utils as comUtils
from d3m.metadata import base as metadata_base
from d3m import metrics
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.unseen_label_encoder import UnseenLabelEncoderPrimitive
from common_primitives.unseen_label_decoder import UnseenLabelDecoderPrimitive
from common_primitives import construct_predictions
from common_primitives import compute_scores

from rpi_d3m_primitives.JMIplus import JMIplus
from rpi_d3m_primitives.JMIplus_auto import JMIplus_auto
from rpi_d3m_primitives.STMBplus_auto import STMBplus_auto
from rpi_d3m_primitives.S2TMBplus import S2TMBplus
from rpi_d3m_primitives.NaiveBayes_PointInf import NaiveBayes_PointInf as NB_P
from rpi_d3m_primitives.NaiveBayes_BayesianInf import NaiveBayes_BayesianInf as NB_B

# Classification
#dataset_name = '38_sick'
#dataset_name = '185_baseball'
#dataset_name = '27_wordLevels' 
#dataset_name = 'uu4_SPECT'  #IPCMB takes time
#dataset_name = '1491_one_hundred_plants_margin'
#dataset_name = '313_spectrometer' 
#dataset_name = '57_hypothyroid'
dataset_name = '4550_MiceProtein' 

# Regerssion
#dataset_name = '26_radon_seed'
#dataset_name = '196_autoMpg'
#dataset_name = '299_libras_move'
#dataset_name = '534_cps_85_wages'

print('\ndataset to dataframe')   
# step 1: dataset to dataframe
path = os.path.join('/Users/zijun/Dropbox/', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\nExtract Attributes')
# step 2: Extract Attributes
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value


# last column will be the true target for evaluation
target_idx = trainD.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']+1

print('\nExtract Targets')
# step 4: extract targets
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

########################################################################################
#print('\nFeature Selection: JMI')
##step 6 feature selection
#hyperparams_class = JMIplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#FSmodel = JMIplus_auto(hyperparams=hyperparams_class.defaults())
#FSmodel.set_training_data(inputs=trainD, outputs=trainL)        
#FSmodel.fit()
#print('\nSelected Feature Index')
#print(FSmodel._index)
#print('\n')

#print('\nFeature Selection: S2TMB')
##step 6 feature selection
#hyperparams_class = S2TMBplus.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#FSmodel = S2TMBplus(hyperparams=hyperparams_class.defaults())
#FSmodel.set_training_data(inputs=trainD, outputs=trainL)        
#FSmodel.fit()
#print('\nSelected Feature Index')
#print(FSmodel._index)
#print('\n')
#

print('\nFeature Selection: STMB')
#step 6 feature selection
hyperparams_class = STMBplus_auto.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
FSmodel = STMBplus_auto(hyperparams=hyperparams_class.defaults())
FSmodel.set_training_data(inputs=trainD, outputs=trainL)        
FSmodel.fit()
print('\nSelected Feature Index')
print(FSmodel._index)
print('\n')

trainD = FSmodel.produce(inputs=trainD) 
trainD = trainD.value


print ('Classification phase: Naive Bayes classifier')
hyperparams_class = NB_B.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
classifier = NB_B(hyperparams=hyperparams_class.defaults())

classifier.set_training_data(inputs=trainD, outputs=trainL)
classifier.fit()
########################################################################################
        

print ('\nLoad testing dataset') 
path = os.path.join('/Users/zijun/Dropbox/', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
#path = '/Users/zijun/Dropbox/38_sick/TEST/dataset_TEST/datasetDoc.json'
dataset = container.Dataset.load('file://{uri}'.format(uri=path))
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\nExtract Attributes')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value


print('\nSubset of testing data')
testD = FSmodel.produce(inputs=testD)
testD = testD.value

print('\nExtract Suggested Target')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

print('\nGet Target Name')
column_metadata = testL.metadata.query((metadata_base.ALL_ELEMENTS, 0))
TargetName = column_metadata.get('name',[])

print('\nClassifier Prediction')
predictedTargets = classifier.produce(inputs=testD)
predictedTargets = predictedTargets.value


print('\nConstruct Predictions')
hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
dataframe = call_metadata.value

print('\ncompute scores')
path = os.path.join('/Users/zijun/Dropbox/', dataset_name, 'SCORE/dataset_TEST/datasetDoc.json')
#path = '/Users/zijun/Dropbox/38_sick/SCORE/dataset_TEST/datasetDoc.json'
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_idx), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_idx), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

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

#groundtruth_path = os.path.join('/Users/zijun/Dropbox/', dataset_name, 'SCORE/targets.csv')
#GT_label = pd.read_csv(groundtruth_path)
#GT_label = container.ndarray(GT_label[TargetName])
#y_pred = predictedTargets.iloc[:,0]
#y_pred = [int(i) for i in y_pred]
#scores = f1_score(GT_label, y_pred, average='macro')

print('\nScore')
print(scores)




