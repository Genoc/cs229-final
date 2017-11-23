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
countyList = ['CHESTER', 'ADAMS', 'BEDFORD', 'ALLEGHENY']
numIterations = 10000
lr = 1e-3

# define model and initial parameters
predictors = ['Party Code','Primary', 'Gender', 'Age'] #'Party Code','Primary', 'Gender', 'Age'
parameterValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
allData = {} 
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

	# drop everything we don't need
	countyData = {} 
	for precinct in precincts:

		# get the people who voted in 2016 in this precinct
		precinctDF = data[(data['Precinct Code'] == precinct) & \
			pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]

		# construct the design matrix and determine the probabilities
		# under the current parameter values 
		designMatrix = util.constructDesignMatrix(predictors, precinctDF, county, countyList)

		# pull the actual trump-clinton vote share
		precinctName = zoneCodes[pd.to_numeric(zoneCodes['Value']) == precinct]['Precinct Name'].values[0]
		trumpVotes = trumpCountyVotes[trumpCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]
		clintonVotes = clintonCountyVotes[clintonCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]

		# store things
		precinctData = {}
		precinctData['Design Matrix'] = designMatrix
		precinctData['Trump Votes'] = trumpVotes
		precinctData['Clinton Votes'] = clintonVotes
		countyData[precinctName] = precinctData
	allData[county] = countyData

# training loop
#print util.computeReferenceLikelihood(allData, parameterValues)
for i in range(numIterations):

	# make periodic updates
	if i % 25 == 0:
		l = util.computeLikelihood(allData, parameterValues)
		with open('trainingPath.txt', 'a') as the_file:
			the_file.write('%s \n' % l)
		print parameterValues
		print l

	# choose a random precinct from a random county
	county = random.choice(allData.keys())
	precinct = random.choice(allData[county].keys())

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

	parameterValues = parameterValues + lr/np.sqrt(1 + i) * (grad1 + grad2) #lr/np.sqrt(1 + i)





