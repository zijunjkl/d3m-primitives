import os
from d3m import container
#from collections import OrderedDict
from d3m.metadata import base as metadata_base
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
#from common_primitives.column_parser import ColumnParserPrimitive
#from common_primitives.unseen_label_encoder import UnseenLabelEncoderPrimitive
#from common_primitives.unseen_label_decoder import UnseenLabelDecoderPrimitive
from common_primitives import dataset_remove_columns
from common_primitives import construct_predictions
from common_primitives import compute_scores
from rpi_d3m_primitives.TreeAugmentedNB_BayesianInf import TreeAugmentedNB_BayesianInf as TAN_B


# Classification
dataset_name = '185_baseball'#remove column 1

print('\nLoad Dataset')   
path = os.path.join('/home/zijun/Dropbox/Project/DARPA-D3M-project/D3Mdatasets-phase1/', dataset_name,'TRAIN/dataset_TRAIN/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

target_index = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))['dimension']['length']-1
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')

print('\nRemove Columns')
remove_columns_hyperparams_class = dataset_remove_columns.RemoveColumnsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
hp = remove_columns_hyperparams_class({'columns': [1], 'resource_id': 'learningData'})
remove_columns_primitive = dataset_remove_columns.RemoveColumnsPrimitive(hyperparams=hp)
dataset = remove_columns_primitive.produce(inputs=dataset).value


print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value


print('\nExtract Attributes')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
trainD = call_metadata.value


print('\nExtract Targets')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
trainL = call_metadata.value

print ('\nLoad testing dataset') 
path = os.path.join('/home/zijun/Dropbox/Project/DARPA-D3M-project/D3Mdatasets-phase1/', dataset_name,'TEST/dataset_TEST/datasetDoc.json')
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Attribute')

print('\nRemove Column')
dataset = remove_columns_primitive.produce(inputs=dataset).value


print('\nDataset to Dataframe')
hyperparams_class = DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = primitive.produce(inputs=dataset)
dataframe = call_metadata.value

print('\nExtract Attributes')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']}))
call_metadata = primitive.produce(inputs=dataframe)
testD = call_metadata.value


print('\nExtract Suggested Target')
hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}))
call_metadata = primitive.produce(inputs=dataframe)
testL = call_metadata.value

print('\nGet Target Name')
column_metadata = testL.metadata.query((metadata_base.ALL_ELEMENTS, 0))
TargetName = column_metadata.get('name',[])


print('\nBayesian Network Classifier')
nbins=7 #bins
N0=8

hyperparams_class =TAN_B.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
classifier =TAN_B(hyperparams=hyperparams_class.defaults().replace({'nbins':nbins,'N0':N0}))
classifier.set_training_data(inputs=trainD, outputs=trainL)
classifier.fit()
predictedTargets = classifier.produce(inputs=testD)
predictedTargets = predictedTargets.value


print('\nConstruct Predictions')
hyperparams_class = construct_predictions.ConstructPredictionsPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
construct_primitive = construct_predictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())
call_metadata = construct_primitive.produce(inputs=predictedTargets, reference=dataframe)
dataframe = call_metadata.value

print('\ncompute scores')
path = os.path.join('/home/zijun/Dropbox/Project/DARPA-D3M-project/D3Mdatasets-phase1/', dataset_name, 'SCORE/dataset_TEST/datasetDoc.json')
#path = '/Users/zijun/Dropbox/38_sick/SCORE/dataset_TEST/datasetDoc.json'
dataset = container.Dataset.load('file://{uri}'.format(uri=path))

#target_idx = dataset.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/Target')
dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, target_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

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
# F1_marco 0.7776

#print('\nSave file')
#os.mkdir('/output/predictions/e7239570-bb9d-464b-aa5b-a0f7be958dc0')
#output_path = os.path.join('/output','predictions','e7239570-bb9d-464b-aa5b-a0f7be958dc0','predictions.csv')
#with open(output_path, 'w') as outputFile:
#    dataframe.to_csv(outputFile, index=False,columns=['d3mIndex', TargetName])
