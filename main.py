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
logLikelihood = 0.0
for countyFile in countyFiles:
	print(logLikelihood)

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

	# pull the relevant county results 
	county = countyFile.replace(' FVE 20171016.txt', '')
	print county
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
	if len(precincts) != len(zoneCodes):
		print 'SOMETHING IS WRONG'
		t()
	zoneCodes = zoneCodes.sort('Precinct Name')
	trumpCountyVotes = trumpCountyVotes.sort('Municipality Name')
	clintonCountyVotes = clintonCountyVotes.sort('Municipality Name')
	trumpCountyVotes['Precinct'] = zoneCodes['Precinct Name'].values
	clintonCountyVotes['Precinct'] = zoneCodes['Precinct Name'].values
	
	for i in range(len(trumpCountyVotes['Precinct'])):	# weak sanity checks
		t1 = trumpCountyVotes['Municipality Name'].values[i]
		t2 = re.sub('[^a-zA-Z ]', '', trumpCountyVotes['Precinct'].values[i]).strip()
		c1 = clintonCountyVotes['Municipality Name'].values[i]
		c2 = re.sub('[^a-zA-Z ]', '', clintonCountyVotes['Precinct'].values[i]).strip()
#		if (t1 != t2) or (c1 != c2):
#			print t1, t2, c1, c2

	# iterate through the precincts
	for precinct in precincts:

		# get the people who voted in 2016 in this precinct
		precinctData = data[(data['Precinct Code'] == precinct) & \
			pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]

		# construct the design matrix and determine the probabilities
		# under the current parameter values 
		designMatrix = constructDesignMatrix(predictors, precinctData)
		probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))

		# pull the actual trump-clinton vote share
		precinctName = zoneCodes[pd.to_numeric(zoneCodes['Value']) == precinct]['Precinct Name'].values[0]
		trumpVotes = trumpCountyVotes[trumpCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]
		clintonVotes = clintonCountyVotes[clintonCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]

		# if the total trump-clinton votes are off from the vote total by more than 10%, yell
#		if abs(probabilities.shape[1]/(trumpVotes + clintonVotes) - 1) > 0.1:
#			print('issue!')

		# get the result
		pb = PoiBin(probabilities.tolist()[0])
		logLikelihood += np.log(max(2e-16, pb.pmf(int(round(trumpVotes/float(trumpVotes + clintonVotes) * probabilities.shape[1])))))


