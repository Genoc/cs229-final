import nltk
import random
import numpy as np
import os
import pdb
import re
import utilities
import collections
import pandas as pd 
from poibin import PoiBin
from pdb import set_trace as t

# read in election results and county mapping 
electionResults = pd.read_csv('../Statewide/20161108__pa__general__precinct.csv',
	header = None)
eColumnNames = pd.read_csv('electionResultsColumnNames.csv', header = None)
electionResults.columns = [i[0] for i in eColumnNames.values.tolist()]
countyMapping = pd.read_csv('countyMapping.csv', header = None)
countyMapping.columns = ['ID', 'County']

# get voter file column names
vfColumnNames = pd.read_csv('voterFileColumnNames.csv', header = None)

# get list of files
arr = os.listdir('../Statewide')
countyFiles = [x for x in arr if 'FVE 20171016.txt' in x]



# define model and initial parameters
predictors = ['Gender', 'Party Code']
parameterValues = [0. for i in range(3)] # fix this


# matrix construction functions 

# flatten function (with thanks to stack overflow)
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def createColumns(predictor, precinctData):
	if predictor == 'Gender':
		return [[1 if x == 'F' else 0 for x in precinctData[predictor]]]
	if predictor == 'Party Code':
		return [[1 if x == 'R' else 0 for x in precinctData[predictor]], 
			[1 if x == 'R' else 0 for x in precinctData[predictor]]]

def constructDesignMatrix(predictors, precinctData):
	vectors = []
	for pred in predictors:
		newCols = createColumns(pred, precinctData)
		for col in newCols:
			vectors.append(col)
	return np.matrix(vectors).T	

# iterate through the files 
for countyFile in countyFiles:

	# read in all the county data as 
	data = pd.read_csv('../Statewide/'+countyFile, sep = '\t', header = None)
	data.columns = [i[0] for i in vfColumnNames.values.tolist()]
	
	# get the election mapping and rename columns
	def getElectionName(q):
		return electionMap['Election Name'][electionMap['Election Number'] == q].values[0]
	def makeReplaceName(currentName):
		r = re.match('Election ([0-9]+)', currentName)
		if r is None:
			return currentName
		else:
			q = int(r.group(1))
			return currentName.replace(r.group(0), getElectionName(q))

	electionMap = pd.read_csv('../Statewide/' + countyFile.replace('FVE', 'Election Map'), 
		sep = '\t', header = None)
	electionMap.columns = ['County', 'Election Number', 'Election Name', 'Election Date']
	data.columns = [makeReplaceName(s) for s in data.columns.values]
	
	# iterate through the precincts
	precincts = np.unique(data['Precinct Code'])
	county = countyFile.replace(' FVE 20171016.txt', '')
	countyCode = countyMapping[countyMapping['County'] == county.title()]['ID'].values[0]
	for precinct in precincts:

		# get the people who voted in 2016 in this precinct
		precinctData = data[(data['Precinct Code'] == precinct) & \
			pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]

		# construct the design matrix and determine the probabilities
		# under the current parameter values 
		designMatrix = constructDesignMatrix(predictors, precinctData)
		probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))

		# pull the actual trump-clinton vote share
		t()