import nltk
import random
import numpy as np
import os
import pdb
import re
import datetime
import utilities as util
import collections
import pandas as pd 
from poibin import PoiBin
from pdb import set_trace as t

# constants
countyList = ['BEDFORD', 'ADAMS', 'CHESTER']#, 'ALLEGHENY']
numIterations = 10000
lr = 1e-3
interceptByCounty = True

# define model and initial parameters
predictors = ['Party Code','Primary', 'Gender', 'Age'] #'Party Code','Primary', 'Gender', 'Age'
if interceptByCounty:
	parameterValues = [0]*(len(countyList) + 2*len(predictors) - (1 if 'Age' in predictors else 0))
else:
	parameterValues = [0]*(1 + 2*len(predictors) - (1 if 'Age' in predictors else 0))


# read in datasets
electionResults = util.readElectionResults('../Statewide/20161108__pa__general__precinct.csv',
	'electionResultsColumnNames.csv')
countyMapping = util.readCountyMapping('countyMapping.csv')
vfColumnNames = pd.read_csv('voterFileColumnNames.csv', header = None)

# get list of files
arr = os.listdir('../Statewide')
def checkMatch(list, string):
	return np.sum([1 if l in string else 0 for l in list])
countyFiles = [x for x in arr if 'FVE 20171016.txt' in x and checkMatch(countyList, x) == 1]

# pre-process for future usage 
allData = util.preProcess(countyFiles, vfColumnNames, countyMapping, \
	electionResults, predictors, countyList, interceptByCounty)

# training loop
#print util.computeReferenceLikelihood(allData, parameterValues)
for i in range(numIterations):

	# choose a random precinct from a random county
	county = random.choice(allData.keys())
	precinct = random.choice(allData[county].keys())

	# make periodic updates
	if i % 20 == 0:
		l = util.computeLikelihood(allData, parameterValues)
		with open('trainingPath.txt', 'a') as the_file:
			the_file.write('%s \n' % l)
		print parameterValues
		print l

	# make a parameter update based on this precicnt and the normal approximation
	designMatrix = allData[county][precinct]['Design Matrix']
	clintonVotes = allData[county][precinct]['Clinton Votes']
	trumpVotes = allData[county][precinct]['Trump Votes']
	probabilities = np.array((1/(1 + np.exp(-designMatrix.dot(parameterValues)))).tolist()[0])
	mu = np.sum(probabilities)
	sigmaSq = np.sum(probabilities*(1-probabilities))

	d = clintonVotes/float(trumpVotes + clintonVotes) * len(probabilities)
	grad1 = np.sum((np.array(designMatrix) * np.expand_dims((d-mu)*probabilities*(1-probabilities), 1)), \
			axis = 0)/sigmaSq

	temp = np.expand_dims(1/2.*((d-mu)**2/sigmaSq**2 - 1/sigmaSq)*(2*probabilities-1)*probabilities**2, axis = 0)
	grad2 = np.sum((np.array(designMatrix)*np.transpose(temp)), axis = 0)
	grad2a = np.sum((np.array(designMatrix) * \
		np.expand_dims(probabilities*(1-probabilities)*(2*probabilities - 1), 1)), \
			axis = 0)/2/sigmaSq

	grad2b = np.sum((np.array(designMatrix) * \
		np.expand_dims(-(d-mu)**2*probabilities*(1-probabilities)*(2*probabilities - 1), 1)), \
			axis = 0)/2/sigmaSq**2


	parameterValues = parameterValues + lr/np.sqrt(1 + i) * (grad1 + grad2a + grad2b) #lr/np.sqrt(1 + i)

	#numGrad = util.computeNumericalGradient(util.computePrecinctLikelihood_normalApprox, allData, parameterValues, \
	#	county, precinct, countyList, interceptByCounty)
	estGrad = grad1 + grad2a + grad2b

	if np.isnan(estGrad).any():
		continue

	parameterValues = parameterValues + lr/np.sqrt(1 + i) * estGrad

