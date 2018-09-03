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
from sklearn.linear_model import LogisticRegression
from poibin import PoiBin
from pdb import set_trace as t

# parse desired training scheme
debug = False
test = True
loadData = True # by default, load data
newDesignMatrices = False # by default don't regenerate design matrces

# constants
if debug:
	countiesToUse = ['BEDFORD', 'CHESTER', 'ARMSTRONG']
else:
	countiesToUse = ['ADAMS', 'ALLEGHENY', 'ARMSTRONG', 'BEAVER', 'BEDFORD',\
	'BLAIR', 'BRADFORD', 'BUTLER', 'CAMBRIA', 'CAMERON', 'CARBON', 'CHESTER',\
	'CLEARFIELD', 'CLINTON', 'COLUMBIA', 'CRAWFORD', 'DELAWARE', 'ERIE', 'FOREST', 'FRANKLIN',\
	'GREENE', 'INDIANA', 'JEFFERSON', 'JUNIATA', 'LAWRENCE', 'LEBANON', 'LUZERNE',\
	'LYCOMING', 'MERCER', 'MIFFLIN', 'MONROE', 'MONTGOMERY', 'MONTOUR', 'McKEAN', 'PERRY',\
	'PHILADELPHIA', 'PIKE', 'POTTER', 'SCHUYLKILL', 'SNYDER', 'SOMERSET', 'SULLIVAN',\
	'SUSQUEHANNA', 'UNION', 'WARREN', 'WASHINGTON', 'WESTMORELAND', 'WYOMING', 'YORK']
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
usableCountyFiles = ['ADAMS FVE 20171016.txt', 'ALLEGHENY FVE 20171016.txt', 'ARMSTRONG FVE 20171016.txt', 'BEAVER FVE 20171016.txt', 'BEDFORD FVE 20171016.txt', 'BLAIR FVE 20171016.txt', 'BRADFORD FVE 20171016.txt', 'BUTLER FVE 20171016.txt', 'CAMBRIA FVE 20171016.txt', 'CAMERON FVE 20171016.txt', 'CARBON FVE 20171016.txt', 'CHESTER FVE 20171016.txt', 'CLEARFIELD FVE 20171016.txt', 'CLINTON FVE 20171016.txt', 'COLUMBIA FVE 20171016.txt', 'CRAWFORD FVE 20171016.txt', 'ERIE FVE 20171016.txt', 'FOREST FVE 20171016.txt', 'FRANKLIN FVE 20171016.txt', 'GREENE FVE 20171016.txt', 'INDIANA FVE 20171016.txt', 'JEFFERSON FVE 20171016.txt', 'JUNIATA FVE 20171016.txt', 'LAWRENCE FVE 20171016.txt', 'LEBANON FVE 20171016.txt', 'LUZERNE FVE 20171016.txt', 'LYCOMING FVE 20171016.txt', 'MERCER FVE 20171016.txt', 'MIFFLIN FVE 20171016.txt', 'MONROE FVE 20171016.txt', 'MONTOUR FVE 20171016.txt', 'McKEAN FVE 20171016.txt', 'PERRY FVE 20171016.txt', 'PHILADELPHIA FVE 20171016.txt', 'PIKE FVE 20171016.txt', 'POTTER FVE 20171016.txt', 'SCHUYLKILL FVE 20171016.txt', 'SNYDER FVE 20171016.txt', 'SOMERSET FVE 20171016.txt', 'SULLIVAN FVE 20171016.txt', 'SUSQUEHANNA FVE 20171016.txt', 'UNION FVE 20171016.txt', 'WARREN FVE 20171016.txt', 'WASHINGTON FVE 20171016.txt', 'WESTMORELAND FVE 20171016.txt', 'WYOMING FVE 20171016.txt', 'YORK FVE 20171016.txt']
countyList = [x.replace(' FVE 20171016.txt','') for x in countyFiles]

# split into training and testing counties
np.random.seed(14)
if not test:
	countyTrain = countyList
else:
	numHoldouts = np.round(0.3*len(countyList))
	holdoutIndices = np.random.choice(range(len(countyList)), int(numHoldouts), replace = False)
	countyTest = [countyList[i] for i in range(len(countyList)) if i in holdoutIndices]
	countyTrain = [countyList[i] for i in range(len(countyList)) if i not in holdoutIndices]

# pre-process for future usage 
if not loadData:
	allData = util.preProcess(countyFiles, vfColumnNames, countyMapping, \
		electionResults, predictors, countyList, interceptByCounty, countyCovariates)
elif newDesignMatrices:
	allData = util.load_allData(newDesignMatrices, vfColumnNames, predictors, countyList, interceptByCounty, countyCovariates)
else:
    allData = util.load_allData()

# aggregate all the matrices
matrices = []
for county in allData.keys():
	for precinct in allData[county]:
		matrices += [allData[county][precinct]['Design Matrix']]
dataMatrix = np.concatenate(matrices)

model = LogisticRegression(C = 1e5)
model.fit(dataMatrix[:,:19], dataMatrix[:,19])
t()
