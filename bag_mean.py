import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

def plot_correlation(df):
	plt.matshow(df.corr().abs())
	plt.colorbar()
	plt.show()


def voting_classifier(estimators, weights):
	return VotingClassifier(estimators = estimators, voting='soft', weights = weights)
	
def support_vector_machine():
	return svm.SVC(kernel = 'rbf', C=10, gamma=0.01, probability = True)
	
def random_forest():
	return RandomForestClassifier()

def logistic_regression():
	return LogisticRegression()

def train(classifier):
	df = pd.read_csv("./data/train.csv")
	#df.iloc[:, 2:164] = df.iloc[:, 2:164].abs()
	
	#print(df)
	#pca = PCA(n_components = 160)
	#pca.fit(df.iloc[:, 2:])
	#print(pd.DataFrame(pca.transform(df.iloc[:, 2:])))
	#plot_correlation(pd.DataFrame(pca.transform(df.iloc[:, 2:])))
	
	plot_correlation(df)
	
	
	bag_means = df.groupby(['molecule']).mean() #for each molecule, create a vector that is the mean of the molecules conformation-vectors
	
	feature_vectors, labels = bag_means.iloc[:, :-1], bag_means.iloc[:, -1:]
	
	X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size = 0.2, stratify = labels)
	
	pipeline = make_pipeline(StandardScaler(), classifier)
	
	cross_val_scores = cross_val_score(pipeline, X_train, np.ravel(y_train), cv = 10)
	print(np.mean(cross_val_scores))
	
	pipeline.fit(X_train, np.ravel(y_train))
	
	y_pred = pipeline.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	
	return classifier
	
def test(classifier):
	df = pd.read_csv("./data/test.csv", index_col = 0)
	
	#df.iloc[:, 2:164] = df.iloc[:, 2:164].abs()
	
	bag_means = df.groupby(['molecule']).mean()
	
	pipeline = make_pipeline(StandardScaler().fit(bag_means), classifier)
	
	pred = pipeline.predict(bag_means)
	
	result_df = pd.DataFrame({
		'molecule': list(bag_means.index),
		'class': list(pred) 
	})
	
	result_df.to_csv('bag_mean_result.csv', index=False)
	
classifier = train(support_vector_machine())

test(classifier)