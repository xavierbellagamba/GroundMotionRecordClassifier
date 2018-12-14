import csv
import numpy as np

##################################################################
#isNumber: check if a string is a number
##################################################################
def isNumber(s):
	try:
		float(s)
		return True

	except ValueError:
		return False


##################################################################
#loadCSV: load the csv data file
##################################################################
def loadCSV(data_path, row_ignore=0, col_ignore=0, isInput=False, isCategorical=False):
	if not isInput:
		M = []
		with open(data_path) as csvfile:
			readCSV = csv.reader(csvfile)

			#Skip header
			for i in range(row_ignore):
				next(csvfile)

			for row in readCSV:
				#Input vector
				single_line = []
				for i in range(col_ignore, len(row)):
					if isNumber(row[i]):
						single_line.append(float(row[i]))
					else:
						single_line.append(row[i])
				M.append(single_line)

		return M

	#input: last column is the output
	elif isInput:
		v_input = []
		v_output = []

		with open(data_path) as csvfile:
			readCSV = csv.reader(csvfile)

			#Skip header
			for i in range(row_ignore):
				next(csvfile)

			for row in readCSV:
				#Input vector
				single_input = []
				for i in range(col_ignore, len(row)-1):
					single_input.append(float(row[i]))
				v_input.append(single_input)
				if isCategorical == True:
					v_output.append(row[-1])
				else:
					if isNumber(row[-1]):
						v_output.append(float(row[-1]))
					else:
						v_output.append(row[-1])
		csvfile.close()

		if isCategorical:
			label = []
			for i in range(len(v_output)):
				if i == 0:
					label.append(v_output[i])
				else:
					isDifferent = True
					for j in range(len(label)):
						if label[j] == v_output[i]:
							isDifferent = False
							break
					if isDifferent:
						label.append(v_output[i])
	
		return [v_input, v_output]


##################################################################
#deskewData: apply transform functions to data
##################################################################
def deskewData(data):
	for i in range(len(data)):
		if i == 0 or i == 1 or i == 11 or i == 15 or i == 16:  
			data[i] = np.log(data[i])
		elif i == 17:
			data[i] = -1.0/data[i]**1.2
		elif i == 2:
			data[i] = data[i]**(-.2)
		elif i == 10:
			data[i] = data[i]**(-.06)
		elif i == 19:
			data[i] = data[i]**.43
		elif i == 7:
			data[i] = data[i]**.1
		elif i == 8:
			data[i] = data[i]**.23
		elif i == 9:
			data[i] = data[i]**.2
		elif i == 18:
			data[i] = data[i]**.33
		elif i == 3:
			data[i] = data[i]**(.05)
		elif i == 5:
			data[i] = data[i]**(.3)
		elif i == 6:
			data[i] = data[i]**(.37)
		elif i == 12:
			data[i] = data[i]**.05
		elif i == 13:
			data[i] = data[i]**.08
		elif i == 4:
			data[i] = data[i]**(.05)
		elif i == 14:
			data[i] = data[i]**(.05)

	return data


##################################################################
#standardizeData: apply transform functions to data
##################################################################
def standardizeData(data, mu, sigma):
	for i in range(len(data)):
		data[i] = (data[i]-mu[i])/sigma[i]

	return data


##################################################################
#decorrelateData: apply Mahalanobis transform on data
##################################################################
def decorrelateData(data, M):
	M = np.array(M)
	data = M.dot(data)
	data = np.transpose(data)

	return data.tolist()


##################################################################
#preprocessData: load and preprocess the data
##################################################################
def preprocessData(M_input, inputFile=None, row_ignore=0, col_ignore=0):
	#Load Mahalanobis matrix
	M = loadCSV('./M.csv')

	#Load scaling parameter (mu, sigma)
	[mu, sigma] = loadCSV('./mu_sigma.csv')

	#Load data
	if len(M_input) == 0:
		M_input = loadCSV(inputFile, row_ignore, col_ignore, isInput=False)
	else:
		M_input = np.asarray(M_input)

	#Deskew, standardize and decorrelate data
	for i in range(len(M_input)):
		M_input[i] = deskewData(M_input[i])
		M_input[i] = standardizeData(M_input[i], mu, sigma)
		M_input[i] = decorrelateData(M_input[i], M)

	return M_input


##################################################################
#sigmoid: compute sigmoid function for each array entry
##################################################################
def sigmoid(v_input):
	v_act = []
	for x in v_input:
		v_act.append(1./(1+np.exp(-x)))
	return v_act


##################################################################
#tanh: compute tanh function for each array entry
##################################################################
def tanh(v_input):
	v_act = []
	for x in v_input:
		v_act.append(np.tanh(x))
	return v_act


##################################################################
#neuralNet: neuralNet class, does not require keras or tensorflow
##################################################################
class neuralNet():
	def __init__(self):
		self.n_input = 0
		self.n_neuron_H1 = 0
		self.n_neuron_H2 = -1
		self.n_output = 0
		self.activation_H1 = 'NA'
		self.activation_H2 = 'NA'
		self.activation_output = 'NA'
		self.w_H1 = []
		self.w_H2 = []
		self.b_H1 = []
		self.b_H2 = []
		self.w_output = []
		self.b_output = []
		
	#loadNN: load and build neural network model
	def loadNN(self, name):
		data_path = './' + name + '/masterF.txt'
		with open(data_path) as masterF:
			readCSV = csv.reader(masterF)

			for row in readCSV:
				if len(row) == 7:
					self.n_input = int(row[0])
					self.n_neuron_H1 = int(row[1])
					self.n_neuron_H2 = int(row[3])
					self.n_output = int(row[5])
					self.activation_H1 = row[2]
					self.activation_H2 = row[4]
					self.activation_output = row[6]
				elif len(row) == 5:
					self.n_input = int(row[0])
					self.n_neuron_H1 = int(row[1])
					self.n_output = int(row[3])
					self.activation_H1 = row[2]
					self.activation_output = row[4]

		masterF.close()

		#Load weights and biases
		#Weights first hidden layer
		data_path = './' + name + '/weight_1.csv'
		self.w_H1 = np.asarray(loadCSV(data_path))
		#Biases first hidden layer
		data_path = './' + name + '/bias_1.csv'
		self.b_H1 = np.asarray(loadCSV(data_path))
		#Weights output layer
		data_path = './' + name + '/weight_output.csv'
		self.w_output = np.asarray(loadCSV(data_path))
		#Biases output layer
		data_path = './' + name + '/bias_output.csv'
		self.b_output = np.asarray(loadCSV(data_path))

		#One hidden layer
		if self.n_neuron_H2 != -1:
			#Weights second hidden layer
			data_path = './' + name + '/weight_2.csv'
			self.w_H2 = np.asarray(loadCSV(data_path))
			#Biases second hidden layer
			data_path = './' + name + '/bias_2.csv'
			self.b_H2 = np.asarray(loadCSV(data_path))
	
	def useNN(self, v_input):
		v_inter = np.array([])
		#Transform input if required
		if isinstance(v_input, list):
			v_input = np.asarray(v_input)

		#First layer
		if self.activation_H1 == 'sigmoid':
			v_inter = sigmoid(np.dot(v_input.T, self.w_H1) + self.b_H1)
		elif self.activation_H1 == 'tanh':
			v_inter = tanh(np.dot(v_input.T, self.w_H1) + self.b_H1)
		else:
			v_inter = np.dot(v_input.T, self.w_H1) + self.b_H1

		#If second layer exist
		if self.n_neuron_H2 != -1:
			if self.activation_H2 == 'sigmoid':
				v_inter = sigmoid(np.dot(v_inter, self.w_H2) + self.b_H2)
			elif self.activation_H2 == 'tanh':
				v_inter = tanh(np.dot(v_inter, self.w_H2) + self.b_H2)
			else:
				v_inter = np.dot(v_inter, self.w_H2) + self.b_H2

		#Final layer
		if self.activation_output == 'sigmoid':
			v_inter = sigmoid(np.dot(v_inter, self.w_output) + self.b_output)
		elif self.activation_output == 'tanh':
			v_inter = tanh(np.dot(v_inter, self.w_output) + self.b_output)
		else:
			v_inter = np.dot(v_inter, self.w_output) + self.b_output

		return v_inter
		
















