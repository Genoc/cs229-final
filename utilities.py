import numpy as np
import scipy
import pandas as pd 
import re
from pdb import set_trace as t
from poibin import PoiBin
from datetime import datetime
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

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
	countyMapping['County'] = countyMapping['County'].str.upper()
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
def createColumns(predictor, precinctData, countyCovariates):
	if predictor == 'Gender':
		return [[1 if x == 'F' else 0 for x in precinctData[predictor]],
		[1 if x == 'M' else 0 for x in precinctData[predictor]]]
	elif predictor == 'Party Code':
		return [[1 if x == 'D' else 0 for x in precinctData[predictor]], 
			[1 if x == 'R' else 0 for x in precinctData[predictor]]]
	elif predictor == 'Primary': 
		return [[1 if x == 'D' else 0 for x in precinctData['2016 GENERAL PRIMARY Party']],
			[1 if x == 'R' else 0 for x in precinctData['2016 GENERAL PRIMARY Party']]]
	elif predictor == 'Age': # super hacky way to scale...
		return [[(2016 - datetime.strptime(x, '%m/%d/%Y').year)/40. for x in precinctData['DOB']]]
	elif predictor == 'Registration Length': # super hacky way to scale...
		return [[(2016 - datetime.strptime(x, '%m/%d/%Y').year)/40. for x in precinctData['Registration Date']]]
	elif predictor == '2012 General':
		return [[1 if x == 'AB' else 0 for x in precinctData['2012 GENERAL ELECTION Vote Method']], 
			[1 if x == 'AP' else 0 for x in precinctData['2012 GENERAL ELECTION Vote Method']]]
	elif predictor == '2014 General':
		return [[1 if x == 'AB' else 0 for x in precinctData['2014 GENERAL ELECTION Vote Method']], 
			[1 if x == 'AP' else 0 for x in precinctData['2014 GENERAL ELECTION Vote Method']]]
	elif predictor == 'Apartment Dweller':
		return [[0 if pd.isnull(x) else 1 for x in precinctData['Apartment Number']]]
	elif predictor == 'County White Percent':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['White.Percent'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]
	elif predictor == 'County Black Percent':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['Black.Percent'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]
	elif predictor == 'County Latino Percent':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['Latino.Percent'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]
	elif predictor == 'County College Educated Percent':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['College.Educated.Percent'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]
	elif predictor == 'County Population Density':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['Population.Density'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]
	elif predictor == 'County Income':
		val = countyCovariates[countyCovariates['County'].values == \
			precinctData['County'].values[0].title()]['Income'].values[0]
		return [[val for _ in range(precinctData.shape[0])]]

# design matrix constructor
def constructDesignMatrix(predictors, precinctData, county, countyList,\
		interceptByCounty, countyCovariates):
	if interceptByCounty:
		vectors = [np.ones(precinctData.shape[0]).tolist() if c == county else \
			np.zeros(precinctData.shape[0]).tolist() for c in countyList]
	else:
		vectors = [np.ones(precinctData.shape[0]).tolist()]
	for pred in predictors:
		newCols = createColumns(pred, precinctData, countyCovariates)
		for col in newCols:
			vectors.append(col)
	return np.matrix(vectors).T	

# functions for remapping election names
def getElectionName(q, electionMap):
	temp = electionMap['Election Name'][electionMap['Election Number'] == q].values
	if(len(temp) == 0):
		return str('Election ID' + str(q))
	return electionMap['Election Name'][electionMap['Election Number'] == q].values[0]

def makeReplaceName(currentName, electionMap):
	r = re.match('Election ([0-9]+)', currentName)
	if r is None:
		return currentName
	else:
		q = int(r.group(1))
		return currentName.replace(r.group(0), getElectionName(q, electionMap))

def load_allData(newDesignMatrices = False, vfColumnNames = None, predictors=None, countyList=None, interceptByCounty=None, countyCovariates = None):    
	allData = pickle.load( open( "allData.pickle", "rb" ) )

	# recalculate if desired
	if (newDesignMatrices):
		for county in countyList:
			countyFile = county + ' FVE 20171016.txt'
			data = readCountyResults('../Statewide/'+countyFile, vfColumnNames)
			for precinctName in allData[county]:
				precinctDF = data[(data['Precinct Code'] == precinct) & \
					pd.notnull(data['2016 GENERAL ELECTION Vote Method'])]
				designMatrix = constructDesignMatrix(predictors, precinctDF, county, countyList, interceptByCounty, countyCovariates)
				allData[county][precinctName]['Design Matrix'] = designMatrix
    
	return allData

# pre process data function
def preProcess(countyFiles, vfColumnNames, countyMapping, electionResults, predictors, countyList,
	interceptByCounty, countyCovariates):
	# loop through the countyfiles
	allData = {}
	for countyFile in countyFiles:

		# get the county name
		county = countyFile.replace(' FVE 20171016.txt', '')

		# read in all the county data 
		data = readCountyResults('../Statewide/'+countyFile, vfColumnNames)
		
		# get the election mapping and rename columns
		electionMap = pd.read_csv('../Statewide/' + countyFile.replace('FVE', 'Election Map'), 
			sep = '\t', header = None)
		electionMap.columns = ['County', 'Election Number', 'Election Name', 'Election Date']
		data.columns = [makeReplaceName(s, electionMap) for s in data.columns.values]
		precincts = np.unique(data['District 1'])

		# pull the relevant county results 
		countyCode = countyMapping[countyMapping['County'] == county.upper()]['ID'].values[0]
		countyElectionResults = electionResults[(electionResults['County Code'] == countyCode) & 
			(electionResults['Candidate Office Code'] == 'USP')]
		trumpCountyVotes = countyElectionResults[countyElectionResults['Candidate Last Name'] == 'TRUMP']
		clintonCountyVotes = countyElectionResults[countyElectionResults['Candidate Last Name'] == 'CLINTON']

		trumpCountyVotes['Municipality Breakdown Name 1'] = trumpCountyVotes['Municipality Breakdown Name 1'].fillna(0)
		trumpCountyVotes['Municipality Breakdown Name 2'] = trumpCountyVotes['Municipality Breakdown Name 2'].fillna(0)
		clintonCountyVotes['Municipality Breakdown Name 1'] = clintonCountyVotes['Municipality Breakdown Name 1'].fillna(0)
		clintonCountyVotes['Municipality Breakdown Name 2'] = clintonCountyVotes['Municipality Breakdown Name 2'].fillna(0)

		trumpCountyVotes['Municipality Type Code'] =  trumpCountyVotes['Municipality Type Code'].apply(pd.to_numeric, errors='coerce')
		trumpCountyVotes['Municipality Breakdown Name 1'] =  trumpCountyVotes['Municipality Breakdown Name 1'].apply(pd.to_numeric, errors='coerce')
		trumpCountyVotes['Municipality Breakdown Name 2'] =  trumpCountyVotes['Municipality Breakdown Name 2'].apply(pd.to_numeric, errors='coerce')
		clintonCountyVotes['Municipality Breakdown Name 1'] =  clintonCountyVotes['Municipality Breakdown Name 1'].apply(pd.to_numeric, errors='coerce')
		clintonCountyVotes['Municipality Breakdown Name 2'] =  clintonCountyVotes['Municipality Breakdown Name 2'].apply(pd.to_numeric, errors='coerce')
    
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
		trumpCountyVotes = trumpCountyVotes.sort_values(['Municipality Name', 'Municipality Type Code', 'Municipality Breakdown Code 1', 'Municipality Breakdown Name 1', 'Municipality Breakdown Name 2'], ascending = [True, False, False, True, True])
		clintonCountyVotes = clintonCountyVotes.sort_values(['Municipality Name', 'Municipality Type Code', 'Municipality Breakdown Code 1', 'Municipality Breakdown Name 1', 'Municipality Breakdown Name 2'], ascending = [True, False, False, True, True])
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
			designMatrix = constructDesignMatrix(predictors, precinctDF, county, countyList,\
				interceptByCounty, countyCovariates)

			# pull the actual trump-clinton vote share
			precinctName = zoneCodes[pd.to_numeric(zoneCodes['Value'], errors='ignore') == precinct]['Precinct Name'].values[0]
			trumpVotes = trumpCountyVotes[trumpCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]
			clintonVotes = clintonCountyVotes[clintonCountyVotes['Precinct'] == precinctName]['Vote Total'].values[0]

			# store things
			precinctData = {}
			precinctData['Design Matrix'] = designMatrix
			precinctData['Trump Votes'] = trumpVotes
			precinctData['Clinton Votes'] = clintonVotes
			countyData[precinctName.strip()] = precinctData
		allData[county] = countyData
	pickle.dump(allData, open( "allData.pickle", "wb" ) )
	return allData

# function to compute the log likelihood for a given county
def computeLikelihood(allData, parameterValues, approx = False, neuralNet = False):

	# iterate through the precincts
	logLikelihood = 0.0
	for county in allData.keys():
		for precinct in allData[county].keys():

			# get all the data
			designMatrix = allData[county][precinct]['Design Matrix']
			clintonVotes = allData[county][precinct]['Clinton Votes']
			trumpVotes = allData[county][precinct]['Trump Votes']
			if not neuralNet:
				probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
				probabilities = [max(p, 0.0) for p in probabilities.tolist()[0]]
			else:
				hidden = designMatrix.dot(parameterValues['W1']) + np.transpose(parameterValues['b1'])
				final = sigmoid(hidden).dot(parameterValues['W2']) + parameterValues['b2']
				probabilities = np.array(sigmoid(final))

			# get the result
			if approx: 
				logLikelihood += computePrecinctLikelihood_normalApprox(parameterValues, \
					allData, county, precinct, neuralNet)
			else: 
				pb = PoiBin(probabilities)
				logLikelihood += np.log(max(2e-16, pb.pmf(int(round(clintonVotes/float(trumpVotes + \
					clintonVotes) * len(probabilities))))))

	return logLikelihood

# function to compute the log likelihood for a given county
def computePrecinctLikelihood(parameterValues, allData, county, precinct):

	# get all the data
	designMatrix = allData[county][precinct]['Design Matrix']
	clintonVotes = allData[county][precinct]['Clinton Votes']
	trumpVotes = allData[county][precinct]['Trump Votes']

	probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
	probabilities = [max(p, 0.0) for p in probabilities.tolist()[0]]

	# get the result
	pb = PoiBin(probabilities)
	return np.log(max(2e-16, pb.pmf(int(round(clintonVotes/float(trumpVotes + \
		clintonVotes) * len(probabilities))))))

# normal approximation
def computePrecinctLikelihood_normalApprox(parameterValues, allData, county, precinct, neuralNet = False):

	# get all the data
	designMatrix = allData[county][precinct]['Design Matrix']
	clintonVotes = allData[county][precinct]['Clinton Votes']
	trumpVotes = allData[county][precinct]['Trump Votes']

	# compute the probabilities 
	if not neuralNet:
		probabilities = np.array(1/(1 + np.exp(-designMatrix.dot(parameterValues))))
		d = round(clintonVotes/float(trumpVotes + clintonVotes) * probabilities.shape[1])
	else:
		hidden = designMatrix.dot(parameterValues['W1']) + np.transpose(parameterValues['b1'])
		final = sigmoid(hidden).dot(parameterValues['W2']) + parameterValues['b2']
		probabilities = np.array(sigmoid(final))
		d = round(clintonVotes/float(trumpVotes + clintonVotes) * probabilities.shape[0])

	mu = np.sum(probabilities)
	sd = np.sqrt(np.sum(probabilities * (1-probabilities)))

	# get the result
	return np.log(max(2e-16, scipy.stats.norm(mu, sd).pdf(d)))

# function to compute the log likelihood for a given county
def computePrecinctLikelihood_binomialApprox(parameterValues, allData, county, precinct):

	# get all the data
	designMatrix = allData[county][precinct]['Design Matrix']
	clintonVotes = allData[county][precinct]['Clinton Votes']
	trumpVotes = allData[county][precinct]['Trump Votes']

	probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
	mu = np.mean(probabilities)
	d = round(clintonVotes/float(trumpVotes + clintonVotes) * probabilities.shape[1])

	# get the result
	return np.log(max(2e-16, scipy.stats.binom(len(probabilities), mu).pmf(d)))


def computeNumericalGradient(function, allData, parameterValues, county, precinct, \
	countyList, interceptByCounty, eps = 1e-4):

	spNumGrad = scipy.optimize.approx_fprime(parameterValues, function, \
		1e-4, allData, county, precinct)
	if interceptByCounty:
		spNumGrad = ([int(c == county) for c in countyList] + \
			[1]*(len(parameterValues) - len(countyList)))*spNumGrad

	return spNumGrad

def printProgress(allData, parameterValues, i, coefficientNames, approx = False):

	# print progress on likelihood
	print('Iteration ' + str(i))
	l = computeLikelihood(allData, parameterValues, approx)
	print('Likelihood: ' + str(l))

	# store training path
	if i == 0:
		with open('trainingPath.txt', 'w') as the_file:
			the_file.write('%s \n' % l)
	else:
		with open('trainingPath.txt', 'a') as the_file:
			the_file.write('%s \n' % l)

	# print parameter values 
	for j in range(len(parameterValues)):
		print(coefficientNames[j] + ': ' + str(parameterValues[j]))

# function to compute the log likelihood for a given county
def computeTestSetStats(allData, parameterValues, countyTest):

	# iterate through the precincts and estimate mean
	clintonVoteTotal = 0.0
	totalVotes = 0
	for county in countyTest:
		for precinct in allData[county].keys():

			# get all the data
			clintonVotes = allData[county][precinct]['Clinton Votes']
			trumpVotes = allData[county][precinct]['Trump Votes']

			clintonVoteTotal += clintonVotes
			totalVotes += clintonVotes + trumpVotes
	clintonVoteMean = clintonVoteTotal/float(totalVotes)

	# now estimate the variance
	clintonPropSumSq = 0.0 
	for county in countyTest:
		for precinct in allData[county].keys():

			# get all the data
			clintonVotes = allData[county][precinct]['Clinton Votes']
			trumpVotes = allData[county][precinct]['Trump Votes']

			clintonPropSumSq += float(clintonVotes + trumpVotes)/totalVotes*\
				(clintonVotes/float(clintonVotes + trumpVotes) - clintonVoteMean)**2
	return (clintonVoteMean, clintonPropSumSq, totalVotes)


# function to compare test set against predictions
def evaluateTestSet(allData, parameterValues, countyTest, clintonPropSumSq, totalVotes,
		name, isFirst, neuralNet = False):

	# iterate through the precincts
	error = 0.0
	for county in countyTest:
		for precinct in allData[county].keys():

			# get all the data
			designMatrix = allData[county][precinct]['Design Matrix']
			clintonVotes = allData[county][precinct]['Clinton Votes']
			trumpVotes = allData[county][precinct]['Trump Votes']

			if not neuralNet:
				probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
				probabilities = [max(p, 0.0) for p in probabilities.tolist()[0]]
			else:
				hidden = sigmoid(designMatrix.dot(parameterValues['W1']) + np.transpose(parameterValues['b1']))
				final = hidden.dot(parameterValues['W2']) + parameterValues['b2']
				probabilities = np.array(sigmoid(final))

			# get the result
			error += float(clintonVotes + trumpVotes)/totalVotes*\
				(np.sum(probabilities)/float(len(probabilities)) -\
				 clintonVotes/float(clintonVotes + trumpVotes))**2

	# store results path
	if isFirst:
		with open('rSquared' + name + '.txt', 'w') as the_file:
			the_file.write('%s \n' % (1.0-error/clintonPropSumSq))
	else:
		with open('rSquared' + name + '.txt', 'a') as the_file:
			the_file.write('%s \n' % (1.0-error/clintonPropSumSq))

def weaklabels(countyList, allData):
	trumpCount = 0
	trumpSize = 0
	hillCount = 0
	hillSize = 0

	weaklabels = {}

	for county in countyList:
		temp_precincts = []
		countyFile = county + ' FVE 20171016.txt'
		for precinctName in allData[county]:
			item = allData[county][precinctName]
			trumpVotes = float(item['Trump Votes'])
			clintonVotes = float(item['Clinton Votes'])
			total = trumpVotes + clintonVotes

			ratio = max(trumpVotes/total, clintonVotes/total)
			if ratio > 0.9:
				item = {}
				item['Precinct Name'] = precinctName.strip()
				# item['Trump Votes'] = trumpVotes
				# item['Clinton Votes'] = clintonVotes

				if (trumpVotes > clintonVotes):
					item['Winner'] = 'Trump'
					trumpCount += 1
					trumpSize += total
					# print('County: ' + county)
				else:
					item['Winner'] = 'Clinton'
					hillCount += 1
					hillSize += total
					#if county != 'ALLEGHENY' and county != 'PHILADELPHIA':
						#print('County: ' + county)

				temp_precincts.append(item)

		weaklabels[county] = temp_precincts

	print('Number of overwhelming Trump precincts: ' + str(trumpCount))
	print('Avg size of overwhelming Trump precinct ' + str(trumpSize/trumpCount))
	print('Number of overwhelming Clinton precincts: ' + str(hillCount))
	print('Avg size of overwhelming Trump precinct ' + str(hillSize/hillCount))
	
	return weaklabels

def evaluteWeakLabels(allData, parameterValues, weakLabelDictionary, iter):

	# iterate through the precincts and estimate mean
	pairs = []
	for county in weakLabelDictionary.keys():
		for item in weakLabelDictionary[county]:

			# get all the data
			precinct = item['Precinct Name']
			if precinct in allData[county].keys():
				designMatrix = allData[county][precinct]['Design Matrix']
			else:
				t()

			probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
			pairs += [(p, 1) if item['Winner'] == 'Clinton' else (p, 0) for p in probabilities.tolist()[0]]
	
	# pull predictions 
	preds = [x[0] for x in pairs]
	y = [x[1] for x in pairs]
	fpr, tpr, threshold = metrics.roc_curve(y, preds)
	roc_auc = metrics.auc(fpr, tpr)

	# plot results
	fig = plt.figure()
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	fig.savefig('./landslideROC' + str(iter) + '.png')

	fig2 = plt.figure()
	bins = np.linspace(0, 1, 100)
	plt.hist([x[0] for x in pairs if x[1] == 0], bins, alpha=0.5, label='R Landslide Precinct Voters', color = 'red', normed = True)
	plt.hist([x[0] for x in pairs if x[1] == 1], bins, alpha=0.5, label='D Landslide Precinct Voters', color = 'blue', normed = True)
	plt.legend(loc='upper center')
	plt.ylabel('Proportion of Voters')
	plt.xlabel('Predicted Probability of Voting Democratic')
	fig2.savefig('./lanslideHist' + str(iter) + '.png')



def evaluatePrimaryVoters(allData, parameterValues, countyTest, iter):

	# iterate through the precincts and estimate mean
	pairs = [] 
	for county in countyTest:
		for precinct in allData[county].keys():

			# get all the data
			des = allData[county][precinct]['Design Matrix']
			designMatrix = np.delete(des, [3,4], axis = 1)


			probabilities = 1/(1 + np.exp(-designMatrix.dot(parameterValues)))
			probabilities = [max(p, 0.0) for p in probabilities.tolist()[0]]

			for i in range(len(probabilities)):
				if des[:,3][i]==1:
					pairs += [(probabilities[i], 1)]
				if des[:,4][i]==1:
					pairs += [(probabilities[i], 0)]

	preds = [x[0] for x in pairs]
	y = [x[1] for x in pairs]
	fpr, tpr, threshold = metrics.roc_curve(y, preds)
	roc_auc = metrics.auc(fpr, tpr)

	# method I: plt
	fig = plt.figure()
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	fig.savefig('./roc' + str(iter) + '.png')

	fig2 = plt.figure()
	bins = np.linspace(0, 1, 100)
	plt.hist([x[0] for x in pairs if x[1] == 0], bins, alpha=0.5, label='Republican Primary Voters', color = 'red')
	plt.hist([x[0] for x in pairs if x[1] == 1], bins, alpha=0.5, label='Democratic Primary Voters', color = 'blue')
	plt.legend(loc='upper center')
	plt.ylabel('Total Voters')
	plt.xlabel('Predicted Probability of Voting Democratic')
	fig2.savefig('./primaryVoters' + str(iter) + '.png')


def printProgress_nn(allData, parameterValues, i, approx = False):

	# print progress on likelihood
	print('Iteration ' + str(i))
	l = computeLikelihood(allData, parameterValues, approx, True)
	print('Likelihood: ' + str(l))

	# store training path
	if i == 0:
		with open('trainingPath.txt', 'w') as the_file:
			the_file.write('%s \n' % l)
	else:
		with open('trainingPath.txt', 'a') as the_file:
			the_file.write('%s \n' % l)


# sigmoid function 
def sigmoid(x):
	"""
	Compute the sigmoid function for the input here.
	"""
	### YOUR CODE HERE

	# stabilization trick thanks to 
	# https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/	
	sx = np.array(x.flatten())

	sx[sx > 0] = 1/(1 + np.exp(-sx[sx > 0]))
	sx[sx < 0] = np.exp(sx[sx < 0])/(1 + np.exp(sx[sx < 0]))
	sx = np.reshape(sx, newshape = x.shape)
	### END YOUR CODE
	return np.matrix(sx)