import numpy as np
from rpi_d3m_primitives.featSelect.discretization import HC_discretization
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

class RelationSet:
	"Container class that holds and modifies training/testing input and output"
	def __init__(self,data,labels=None,discrete_flag=False):
		self.data = data
		self.labels = labels
		self.discrete_flag = discrete_flag
		self.NUM_STATES = 10 #used for entropy-based discretization
		self.removeIdx = []
		self.optimal_split = []
		self.num_features = data.shape[1]

	def subset(self,indices):
		data = self.data[:,indices]
		return RelationSet(data,self.labels,discrete_flag=self.discrete_flag)

	def split(self,split_point):
		data1 = self.data[:split_point,:]
		data2 = self.data[split_point:,:]
		labels1 = self.labels[:split_point]
		labels2 = self.labels[split_point:]
		rs1 = RelationSet(data1,labels1,discrete_flag=self.discrete_flag)
		rs2 = RelationSet(data2,labels2,discrete_flag=self.discrete_flag)
		return rs1, rs2

	def getStateNo(self, cate_flag=None, n_bins=None):
		stateNo = []
		removeIdx = []
		for i in range(self.num_features):
			num = len(np.unique(self.data[:, i]))
			if num == 1:
				removeIdx.append(i)#for features with only one value, remove
			if not len(cate_flag) == 0:
				if cate_flag[i] == 0:
					num = n_bins
			stateNo.append(num)

		self.removeIdx = removeIdx
		self.NUM_STATES = np.array(stateNo)


	def discretize(self):
		raw_data = self.data
		raw_labels = self.labels
		data,labels,split = HC_discretization(raw_data,raw_labels,self.NUM_STATES)
		self.data = np.array(data).astype(int)
		self.labels = np.array(labels).astype(int)
		stateNo = []
		removeIdx = []
		for i in range(self.num_features):
			num = len(np.unique(self.data[:,i]))
			if num == 1:
				removeIdx.append(i)
			stateNo.append(num)

		self.removeIdx = removeIdx
		self.NUM_STATES = np.array(stateNo)
		self.discrete_flag = True


	def impute(self):
		raw_data = self.data
		if not np.count_nonzero(np.isnan(raw_data)) == 0:  # if there is missing values
			for i in range(self.num_features):  # if all features for the feature is nan
				if raw_data.shape[0] == np.count_nonzero(np.isnan(raw_data[:,i])):
					raw_data[:,i] = np.zeros(raw_data.shape[0],)
		if not np.count_nonzero(np.isnan(raw_data)) == 0:  # if still having missing values
			imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
			imp.fit(raw_data)
			raw_data = imp.transform(raw_data)
		self.data = raw_data


	def resample(self,start,end):
		data = self.data[start:end,:]
		labels = self.labels[start:end]
		rs = RelationSet(data,labels,discrete_flag=self.discrete_flag)
		return rs
    
	def remove(self):
		index_list = np.setdiff1d(np.arange(self.num_features), np.array(self.removeIdx))
		self.NUM_STATES = self.NUM_STATES[index_list]
		self.data = self.data[:,index_list]
		#self.num_features = len(self.NUM_STATES)
