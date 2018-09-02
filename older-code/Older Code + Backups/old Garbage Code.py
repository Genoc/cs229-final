# FUCK
intercept = [-0.4, -0.2, 0.0, 0.2, 0.4]
dem = [x/5.0 for x in range(5)]
rep = [-x/5.0 for x in range(5)]
for i in intercept:
	for d in dem:
		for r in rep:
			print i, d, r, util.computeLikelihood(allData, [i, d, r])





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