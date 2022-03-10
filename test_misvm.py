import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import copy as cp

import misvm

class misvm_extension(misvm.STK):
	def predict(self, arg):
		return np.sign(super().predict(arg))

class bag_scaler:
	def __init__(self):
		self.mean = None
		self.std = None
	
	def fit(self, X, y = None, sample_weight = None):
		feature_vectors = np.vstack(X)
		
		self.mean = np.mean(feature_vectors, axis = 0)
		self.std = np.std(feature_vectors, axis = 0)
		
		return self
	
	def transform(self, X, copy = None):
		arr = cp.deepcopy(X)
		
		for bag in arr:
			for feature_vector in bag:
				feature_vector[:] = np.apply_along_axis(lambda v: (v - self.mean) / self.std, axis = 0, arr = feature_vector)
		
		return arr
	
def create_bags(df):
	molecules = list(dict.fromkeys(df.index))
	
	bags = []
	
	for index, molecule in enumerate(molecules):
		bag = np.array(df.loc[molecule, :], dtype = 'float') # select all rows with the same row name
		
		bags.append(bag)
	
	return (molecules, bags)

def prepare_training_data(x, y):
	labels, x_bags = create_bags(x) # create bags for the features
	labels, y_bags = create_bags(y) # create bags for the classes
		
	classes = list(map(lambda bag: 1 if 1 in bag else -1, y_bags)) # determine target class for each class-bag
	
	return (x_bags, classes)

def train():
	df = pd.read_csv("./data/train.csv", index_col=0)
	
	x, y = df.iloc[:, 1:-1], df.iloc[:, -1:] # separate features from class(features in x, class in y)
	
	feature_bags, classes = prepare_training_data(x, y) # create the feature bags and classes for the training data
	feature_bags, classes = np.array(feature_bags, dtype = 'object'), classes
	feature_bags, classes = shuffle(feature_bags, classes) 
	
	train_bags, test_bags, train_classes, test_classes = train_test_split(feature_bags, classes, test_size = 0.2, stratify = classes)
	
	classifier = misvm_extension(kernel = 'quadratic', C = 0.1)
	
	#classifier = GridSearchCV(classifier, {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['quadratic']}, refit = True)
	
	pipeline = make_pipeline(bag_scaler(), classifier)
	
	validation_scores = cross_val_score(pipeline, train_bags, train_classes, cv = StratifiedKFold(n_splits=10, shuffle=True))
	print("CROSS VALIDATION RESULTS")
	print(validation_scores)
	print("CROSS VALIDATION MEAN: " + str(np.mean(validation_scores)))	
	
	pipeline.fit(train_bags, train_classes)
	
	#print(classifier.best_params_)
	
	pred = pipeline.predict(test_bags)
	print(pred)
	print(test_classes)
	print("\nTEST SET ACCURACY")
	print(accuracy_score(test_classes, pred))
	
	return classifier #pipeline

def test(classifier):
	df = pd.read_csv("./data/test.csv", index_col = 0)
	
	x = df.iloc[:, 1:]
	
	molecules, x_bags = create_bags(x)
	x_bags = np.array(x_bags, dtype = 'object')
	
	scaler = bag_scaler().fit(x_bags)
	x_bags = scaler.transform(x_bags)
	
	y_pred = classifier.predict(x_bags)
	
	result_df = pd.DataFrame({
		'molecule': list(molecules),
		'class': list((y_pred + 1) / 2) # convert classes from (1, -1) to (1, 0)
	})
	
	result_df.to_csv('results.csv', index=False)
	
	
	
classifier = train()

test(classifier)
