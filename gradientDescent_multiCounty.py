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
import sys
from poibin import PoiBin
from pdb import set_trace as t

# parse desired training scheme
stochasticGD = True
debug = False
test = True
loadData = False # by default, generate data from scratch
newDesignMatrices = False # by default don't regenerate design matrces
regularize = False 
if 'batch' in sys.argv:
	stochasticGD = False
if 'debug' in sys.argv:
	debug = True 
if 'trainOnly' in sys.argv:
	test = False
if 'loadData' in sys.argv:
	loadData = True
if 'newDesignMatrices' in sys.argv:
	newDesignMatrices = True
if 'regularize' in sys.argv:
	regularize = True 
	lam = 10 

# constants
if debug:
	countiesToUse = ['BEDFORD', 'CHESTER', 'ARMSTRONG']
else:
	countiesToUse = ['ADAMS', 'ALLEGHENY', 'ARMSTRONG', 'BEAVER', 'BEDFORD',\
	'BLAIR', 'BRADFORD', 'BUTLER', 'CAMBRIA', 'CAMERON', 'CARBON', 'CHESTER',\
	'CLEARFIELD', 'CLINTON', 'COLUMBIA', 'CRAWFORD', 'ERIE', 'FOREST', 'FRANKLIN',\
	'GREENE', 'INDIANA', 'JEFFERSON', 'JUNIATA', 'LAWRENCE', 'LEBANON', 'LUZERNE',\
	'LYCOMING', 'MERCER', 'MIFFLIN', 'MONROE', 'MONTOUR', 'McKEAN', 'PERRY',\
	'PHILADELPHIA', 'PIKE', 'POTTER', 'SCHUYLKILL', 'SNYDER', 'SOMERSET', 'SULLIVAN',\
	'SUSQUEHANNA', 'UNION', 'WARREN', 'WASHINGTON', 'WYOMING', 'YORK']
lr = 1e-3 if stochasticGD else 1e-4
interceptByCounty = False 

# define model and initial parameters
predictors = ['Party Code','Primary', 'Gender', 'Age', '2012 General', '2014 General', 'Apartment Dweller',\
	'County White Percent', 'County Black Percent', 'County Latino Percent', 'County College Educated Percent',\
	'County Population Density', 'County Income'] 
predictorMetaData = {'Party Code': {'len': 2, 'names': ['Registered Dem', 'Registered Rep']}, 
					 'Primary': {'len': 2, 'names': ['Primary Dem', 'Primary Rep']},
					 'Gender': {'len': 2, 'names': ['Female', 'Male']},
					 'Age': {'len': 1, 'names': ['Age']},
					 '2012 General': {'len': 2, 'names': ['2012 Voted Absentee', '2012 Voted In Person']},
					 '2014 General': {'len': 2, 'names': ['2014 Voted Absentee', '2014 Voted In Person']},
					 'Apartment Dweller': {'len': 1, 'names': ['Apartment Dweller']}, 
					 'County White Percent': {'len': 1, 'names': ['County White Percent']},
					 'County Black Percent': {'len': 1, 'names': ['County Black Percent']},
					 'County Latino Percent': {'len': 1, 'names': ['County Latino Percent']},
					 'County College Educated Percent': {'len': 1, 'names': ['County College Educated Percent']},
					 'County Population Density': {'len': 1, 'names': ['County Population Density']},
					 'County Income': {'len': 1, 'names': ['County Income']}}

if interceptByCounty:
	parameterValues = [0]*(len(countiesToUse) + np.sum([predictorMetaData[i]['len'] for i in predictors]))
	coefficientNames = countiesToUse + [p for i in predictors for p in predictorMetaData[i]['names']]
else: 
	parameterValues = [0]*(1 + np.sum([predictorMetaData[i]['len'] for i in predictors]))
	coefficientNames = ['Intercept'] + [p for i in predictors for p in predictorMetaData[i]['names']]

# read in datasets
electionResults = util.readElectionResults('../Statewide/20161108__pa__general__precinct.csv',
	'electionResultsColumnNames.csv')
countyCovariates = pd.read_csv('./Demographics By County_scaled.csv')
countyMapping = util.readCountyMapping('countyMapping.csv')
vfColumnNames = pd.read_csv('voterFileColumnNames.csv', header = None)


# get list of files
arr = os.listdir('../Statewide')
def checkMatch(list, string):
	return np.sum([1 if l in string else 0 for l in list])
countyFiles = [x for x in arr if 'FVE 20171016.txt' in x and checkMatch(countiesToUse, x) == 1]
usableCountyFiles = ['ADAMS FVE 20171016.txt', 'ALLEGHENY FVE 20171016.txt', 'ARMSTRONG FVE 20171016.txt', 'BEAVER FVE 20171016.txt', 'BEDFORD FVE 20171016.txt', 'BLAIR FVE 20171016.txt', 'BRADFORD FVE 20171016.txt', 'BUTLER FVE 20171016.txt', 'CAMBRIA FVE 20171016.txt', 'CAMERON FVE 20171016.txt', 'CARBON FVE 20171016.txt', 'CHESTER FVE 20171016.txt', 'CLEARFIELD FVE 20171016.txt', 'CLINTON FVE 20171016.txt', 'COLUMBIA FVE 20171016.txt', 'CRAWFORD FVE 20171016.txt', 'ERIE FVE 20171016.txt', 'FOREST FVE 20171016.txt', 'FRANKLIN FVE 20171016.txt', 'GREENE FVE 20171016.txt', 'INDIANA FVE 20171016.txt', 'JEFFERSON FVE 20171016.txt', 'JUNIATA FVE 20171016.txt', 'LAWRENCE FVE 20171016.txt', 'LEBANON FVE 20171016.txt', 'LUZERNE FVE 20171016.txt', 'LYCOMING FVE 20171016.txt', 'MERCER FVE 20171016.txt', 'MIFFLIN FVE 20171016.txt', 'MONROE FVE 20171016.txt', 'MONTOUR FVE 20171016.txt', 'McKEAN FVE 20171016.txt', 'PERRY FVE 20171016.txt', 'PHILADELPHIA FVE 20171016.txt', 'PIKE FVE 20171016.txt', 'POTTER FVE 20171016.txt', 'SCHUYLKILL FVE 20171016.txt', 'SNYDER FVE 20171016.txt', 'SOMERSET FVE 20171016.txt', 'SULLIVAN FVE 20171016.txt', 'SUSQUEHANNA FVE 20171016.txt', 'UNION FVE 20171016.txt', 'WARREN FVE 20171016.txt', 'WASHINGTON FVE 20171016.txt', 'WYOMING FVE 20171016.txt', 'YORK FVE 20171016.txt']
countyFiles = set(countyFiles).intersection(usableCountyFiles)
countyList = [x.replace(' FVE 20171016.txt','') for x in countyFiles]

# split into training and testing counties
np.random.seed(2017)
if not test:
	countyTrain = countyList
else:
	numHoldouts = np.round(0.3*len(countyList))
	holdoutIndices = np.random.choice(range(len(countyList)), int(numHoldouts), replace = False)
	countyTest = [countyList[i] for i in range(len(countyList)) if i in holdoutIndices]
	countyTrain = [countyList[i] for i in range(len(countyList)) if i not in holdoutIndices]

# pre-process for future usage 
if loadData==False:
	allData = util.preProcess(countyFiles, vfColumnNames, countyMapping, \
		electionResults, predictors, countyList, interceptByCounty, countyCovariates)
else:
	allData = util.load_allData(newDesignMatrices, predictors, countyList, interceptByCounty, countyCovariates)

# training loop
numIterations = 10000
if stochasticGD:
	for i in range(numIterations):

		# choose a random precinct from a random county
		county = random.choice(countyTrain)
		precinct = random.choice(allData[county].keys())

		# make periodic updates
		if i % 500 == 0:
			util.printProgress(allData, parameterValues, i, predictors, predictorMetaData)

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
		estGrad = grad1 + grad2a + grad2b

		if np.isnan(estGrad).any():
			continue

		parameterValues = parameterValues + lr/np.sqrt(1 + i) * estGrad
else: 
	for i in range(numIterations):

		# report current likelihood
		if i % 20 == 0:
			util.printProgress(allData, parameterValues, i, coefficientNames)
		util.evaluateTestSet(allData, parameterValues, countyTest)

		# make a parameter update after seeing all precincts
		grad = [0.] * len(parameterValues)
		for county in countyTrain:
			for precinct in allData[county].keys():
				designMatrix = allData[county][precinct]['Design Matrix']
				clintonVotes = allData[county][precinct]['Clinton Votes']
				trumpVotes = allData[county][precinct]['Trump Votes']
				probabilities = np.array((1/(1 + np.exp(-designMatrix.dot(parameterValues)))).tolist()[0])
				mu = np.sum(probabilities)
				sigmaSq = np.sum(probabilities*(1-probabilities))

				d = clintonVotes/float(trumpVotes + clintonVotes) * len(probabilities)
				grad1 = np.sum((np.array(designMatrix) * np.expand_dims((d-mu)*probabilities*(1-probabilities), 1)), \
						axis = 0)/sigmaSq

				# components of the gradients 
				temp = np.expand_dims(1/2.*((d-mu)**2/sigmaSq**2 - 1/sigmaSq)*(2*probabilities-1)*probabilities**2, axis = 0)
				grad2 = np.sum((np.array(designMatrix)*np.transpose(temp)), axis = 0)
				grad2a = np.sum((np.array(designMatrix) * \
					np.expand_dims(probabilities*(1-probabilities)*(2*probabilities - 1), 1)), \
						axis = 0)/2/sigmaSq

				grad2b = np.sum((np.array(designMatrix) * \
					np.expand_dims(-(d-mu)**2*probabilities*(1-probabilities)*(2*probabilities - 1), 1)), \
						axis = 0)/2/sigmaSq**2

				# compute the gradient 
				grad = grad1 + grad2a + grad2b
				if regularize:
					grad -= lam*np.array(parameterValues)

				parameterValues = parameterValues + lr/np.sqrt(1 + i) * (grad1 + grad2a + grad2b) #lr/np.sqrt(1 + i)
				estGrad = grad1 + grad2a + grad2b

				if np.isnan(estGrad).any():
					continue

				grad += estGrad

		parameterValues = parameterValues + lr * estGrad