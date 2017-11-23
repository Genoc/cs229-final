import nltk
import random
import numpy as np
import os
import pdb
import re
import utilities as util
import collections
import pandas as pd 
from poibin import PoiBin
from pdb import set_trace as t

# constants
filter = 'ALLEGHENY'
numIterations = 10000
lr = 1e-5

# define model and initial parameters
predictors = ['Party Code', 'Primary', 'Gender', 'Age'] #'Gender', 'Age', 
parameterValues = [0. for i in range(8)] # fix this

# read in datasets
electionResults = util.readElectionResults('../Statewide/20161108__pa__general__precinct.csv',
	'electionResultsColumnNames.csv')
countyMapping = util.readCountyMapping('countyMapping.csv')
vfColumnNames = pd.read_csv('voterFileColumnNames.csv', header = None)

# get list of files
arr = os.listdir('../Statewide')
countyFiles = [x for x in arr if 'FVE 20171016.txt' in x]
if filter is not None:
	countyFiles = [i for i in countyFiles if filter in i]

# iterate through the filePs 
for countyFile in countyFiles:

	# get the county name
	county = countyFile.replace(' FVE 20171016.txt', '')

	# read in all the county data 
	data = util.readCountyResults('../Statewide/'+countyFile, vfColumnNames)
	
	# get the election mapping and rename columns
	electionMap = pd.read_csv('../Statewide/' + countyFile.replace('FVE', 'Election Map'), 
		sep = '\t', header = None)
	electionMap.columns = ['County', 'Election Number', 'Election Name', 'Election Date']
	data.columns = [util.makeReplaceName(s, electionMap) for s in data.columns.values]
	precincts = np.unique(data['District 1'])

	# pull the relevant county results 
	countyCode = countyMapping[countyMapping['County'] == county.title()]['ID'].values[0]
	countyElectionResults = electionResults[(electionResults['County Code'] == countyCode) & 
		(electionResults['Candidate Office Code'] == 'USP')]
	trumpCountyVotes = countyElectionResults[countyElectionResults['Candidate Last Name'] == 'TRUMP']
	clintonCountyVotes = countyElectionResults[countyElectionResults['Candidate Last Name'] == 'CLINTON']

	# pull the precinct mapping
	zoneCodes = pd.read_csv('../Statewide/' + countyFile.replace('FVE', 'Zone Codes'), 
		sep = '\t', header = None)
	zoneCodes.columns = ['County', 'Column', 'Value', 'Precinct Name']
	zoneCodes = zoneCodes[zoneCodes['Column'] == 1]

	# THIS IS AN EXTREMELY HACKY WAY TO TRY TO PAPER OVER SOME OF THE #
	# MAPPING ISSUES AND I HAVE NO IDEA IF IT WILL WORK BUT OH WELL   #
	precincts = np.unique(data['District 1'])
	if len(precincts) != len(trumpCountyVotes):
		print 'SOMETHING IS WRONG'
		t()
	zoneCodes = zoneCodes.sort_values(by = 'Precinct Name')
	trumpCountyVotes = trumpCountyVotes.sort_values(by = 'Municipality Name')
	clintonCountyVotes = clintonCountyVotes.sort_values(by = 'Municipality Name')
	trumpCountyVotes['Precinct'] = zoneCodes['Precinct Name'].values
	clintonCountyVotes['Precinct'] = zoneCodes['Precinct Name'].values


	# parameter updates 
	for i in range(numIterations):

		if i % 20 == 0:
			print parameterValues
		if i % 1000 == 0:
			logLikelihood = util.computeCountyLikelihood(countyFile, vfColumnNames,
				countyMapping, county, electionResults, predictors, parameterValues)
			print logLikelihood
			print parameterValues
		
		# sample a precinct at random
		precinct = random.choice(precincts)

		# get the people who voted in 2016 in this precinct
		precinctData = data[(data['Precinct Code'] == precinct) & \
			pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]

		# construct the design matrix and determine the probabilities
		# under the current parameter values 
		designMatrix = util.constructDesignMatrix(predictors, precinctData)
		probabilities = np.array((1/(1 + np.exp(-designMatrix.dot(parameterValues)))).tolist()[0])

		# pull the actual trump-clinton vote share
		precinctName = zoneCodes[pd.to_numeric(zoneCodes['Value']) == precinct]['Precinct Name'].values[0]
		trumpVotes = trumpCountyVotes[trumpCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]
		clintonVotes = clintonCountyVotes[clintonCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]

		# make a parameter update based on this precicnt and the normal approximation
		mu = np.sum(probabilities)
		d = clintonVotes/float(trumpVotes + clintonVotes) * len(probabilities)
		grad = np.sum((np.array(designMatrix) * np.expand_dims((d-mu)*probabilities*(1-probabilities), 1)), axis = 0)
		parameterValues = parameterValues + lr/(1 + i) * grad
