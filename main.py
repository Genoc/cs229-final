import nltk
import random
import numpy as np
import os
import pdb
import re
import utilities
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

def constructDesignMatrix(predictors, precinctData):
	t()


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
	for precinct in precincts:
		precinctData = data[(data['Precinct Code'] == precinct) & \
			pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]
		constructDesignMatrix(predictors, precinctData)



# [getElectionName(re.match('Election ([0-9]+)', s).group(1)) for s in data.columns.values if re.match('Election', s)]
#	map(data.columns[69:], lambda x: )
# electionMap['Election Name'][electionMap['Election Number'] == 37]
# 	getElectionName(int(re.match('Election ([0-9]+)', s).group(1))) \

