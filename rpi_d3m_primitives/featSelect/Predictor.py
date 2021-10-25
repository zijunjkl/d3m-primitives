from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import clone
from sklearn.metrics import f1_score
import numpy as np

#----------------------- PREDICTOR -----------------------
class Predictor:
	""" Adds functionality to the sklearn predictor base """ 
	def __init__(self,model,testing_set,training_set,tolerance=0.001):
		self.unfit_model = model
		self.testing_set = testing_set
		self.training_set = training_set
		#'tolerance' represents the smallest difference in accuracy considered meaningful. 
		#	If two independence thresholds yield predicitions within 'tolerance,' the 
		#	better one is the one with less features
		self.tolerance = tolerance

	def predict(self,selected_feats):
		train_data = self.training_set.data
		train_labels = self.training_set.labels
		test_data = self.testing_set.data
		test_labels = self.testing_set.labels

		predictor = clone(self.unfit_model)
		predictor.fit(train_data[:,selected_feats],train_labels[:,0])
		return predictor.predict(test_data[:,selected_feats])

	def score_from_labels(self,predictions):
		pass

	def score(self,selected_feats,cache):
		if (selected_feats.tostring() in cache):
#			score = cache[selected_feats.tostring()]
			predictions = self.predict(selected_feats)
			score = self.score_from_labels(predictions)
		else:	
			predictions = self.predict(selected_feats)
			score = self.score_from_labels(predictions)
			cache[selected_feats.tostring()] = score
		return score

	def compare_scores(self,left_score,right_score):
		diff = self.score_difference(left_score,right_score)
		flag = abs(diff) > self.tolerance
		return diff,flag	

	def left_performs_better(self,left_score,left_feats,right_score,right_feats):
		"""	Returns true if the first argument is a better prediction, based on accuracy/mse,
				then the number of selected features
		"""
		"Uses MSE for regression problems and accuracy for classification"
		test_labels = self.testing_set.labels
		score_difference,significant_flag = self.compare_scores(left_score,right_score)
		if (significant_flag):
			return score_difference > 0
		return left_feats.size < right_feats.size

	def choose(self,left_feats,right_feats,optimal_feats,cache):
		"""	Returns the better prediction and feature count, as determined by the metric of 
				'left_performs_better'
		"""
		# Handling for empty feature sets
		if (left_feats.size==0 or right_feats.size==0):
			if (left_feats.size != 0):
				return left_feats,self.score(left_feats,cache)
			if (right_feats.size != 0):
				return right_feats,self.score(right_feats,cache)
			# already in cache
			return optimal_feats,self.score(optimal_feats,cache)

		left_score = self.score(left_feats,cache)
		right_score = self.score(right_feats,cache)
		#print(left_score, right_score)
		if (self.left_performs_better(left_score,left_feats,right_score,right_feats)):
			return left_feats,left_score
		return right_feats,right_score


class Classifier(Predictor):
	def score_difference(self,left_score,right_score):
		return left_score - right_score

	def score_from_labels(self,predictions):
		score = f1_score(self.testing_set.labels, predictions, average='macro')
		return score
#		return accuracy_score(self.testing_set.labels,predictions)

class Regressor(Predictor):
	def score_difference(self,left_score,right_score):
		return right_score - left_score

	def score_from_labels(self,predictions):
		return mean_squared_error(self.testing_set.labels,predictions)