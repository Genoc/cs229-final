import numpy as np
import pandas as pd 
import re
from pdb import set_trace as t
from poibin import PoiBin
from datetime import datetime


# read in election results
def readElectionResults(path, colNamesPath):
	electionResults = pd.read_csv(path, header = None)
	eColumnNames = pd.read_csv(colNamesPath, header = None)
	electionResults.columns = [i[0] for i in eColumnNames.values.tolist()]
	return electionResults

# read in county mapipng 
def readCountyMapping(path):
	countyMapping = pd.read_csv(path, header = None)
	countyMapping.columns = ['ID', 'County']
	return countyMapping

# read in county results 
def readCountyResults(path, colNames):
	d = pd.read_csv(path, sep = '\t', header = None)
	d.columns = [i[0] for i in colNames.values.tolist()]
	return d

# flatten function (with thanks to stack overflow)
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

# create design matrix columns
def createColumns(predictor, precinctData):
	if predictor == 'Gender':
		return [[1 if x == 'F' else 0 for x in precinctData[predictor]],
		[1 if x == 'M' else 0 for x in precinctData[predictor]]]
	elif predictor == 'Party Code':
		return [[1 if x == 'D' else 0 for x in precinctData[predictor]], 
			[1 if x == 'R' else 0 for x in precinctData[predictor]]]
	elif predictor == 'Age': # super hacky way to scale...
		return [[(2016 - datetime.strptime(x, '%m/%d/%Y').year)/40 for x in precinctData['DOB']]]
	elif predictor == 'Primary': # super hacky way to scale...
		return [[1 if x == 'D' else 0 for x in precinctData['2016 GENERAL PRIMARY Party']],
			[1 if x == 'R' else 0 for x in precinctData['2016 GENERAL PRIMARY Party']]]

# design matrix constructor
def constructDesignMatrix(predictors, precinctData):
	vectors = [np.ones(precinctData.shape[0]).tolist()]
	for pred in predictors:
		newCols = createColumns(pred, precinctData)
		for col in newCols:
			vectors.append(col)
	return np.matrix(vectors).T	

# functions for remapping election names
def getElectionName(q, electionMap):
	return electionMap['Election Name'][electionMap['Election Number'] == q].values[0]
def makeReplaceName(currentName, electionMap):
	r = re.match('Election ([0-9]+)', currentName)
	if r is None:
		return currentName
	else:
		q = int(r.group(1))
		return currentName.replace(r.group(0), getElectionName(q, electionMap))

# function to compute the log likelihood for a given county
def computeCountyLikelihood(countyFile, vfColumnNames, countyMapping, county,
	electionResults, predictors, parameterValues):
	# read in all the county data 
	data = readCountyResults('../Statewide/'+countyFile, vfColumnNames)
	
	# get the election mapping and rename columns
	electionMap = pd.read_csv('../Statewide/' + countyFile.replace('FVE', 'Election Map'), 
		sep = '\t', header = None)
	electionMap.columns = ['County', 'Election Number', 'Election Name', 'Election Date']
	data.columns = [makeReplaceName(s, electionMap) for s in data.columns.values]
	
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

	# iterate through the precincts
	logLikelihood = 0.0
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

		# get the result
		pb = PoiBin(probabilities.tolist()[0])
		logLikelihood += np.log(max(2e-16, pb.pmf(int(round(clintonVotes/float(trumpVotes + \
			clintonVotes) * probabilities.shape[1])))))
	return logLikelihood




