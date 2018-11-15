#############################
#       MINI Project 1      #
#         lstm model        #
# Version 8                 #
# Date 22/04/2018           #
# Author: manguy            #
#############################
#%%timeit

import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dlc_bci as bci

#####################
#  MODEL DEFINITION #
#####################

class LSTM(nn.Module):

	def __init__(self,input_size,hidden_size,output_size,nb_layer=2):
		super(LSTM, self).__init__()
		#nb layer defines the number of lstm layers in the model
		self.nb_layer = nb_layer
		#hidden_size defines the hidden size between lstm and the linear layer
		#to output
		self.hidden_size = hidden_size

		#hidden need to be initilized to zeros
		self.hidden = self.initHidden()

		#lstm layer
		self.lstm=nn.LSTM(input_size,hidden_size,nb_layer)
		#self.lstm=nn.LSTM(input_size,hidden_size,nb_layer, dropout=0.05)

		#output layer from hidden to output (size = 2)
		self.hidden2output = nn.Linear(hidden_size, output_size)

		#batch normalization to improves performance
		self.batchnorm = nn.BatchNorm1d(self.hidden_size)

		#LogSoftmax to produce two-class classification probabilities
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, input):
		#First transform the input in order to make the dimension match with the
		#lstm model
		input = input.transpose(2,0).transpose(1,2)
		
		#The update the self.hidden from the outuput of lstm
		output,self.hidden=self.lstm(input,self.hidden)
		
		#Batch normalization to improves performance
		#the output taken is the last one, the one that
		#correponds to the whole sequence
		y = self.batchnorm(output[-1])

		#Produce ouput of size 2
		y = self.hidden2output(y)

		#Use of dropout to prevent overfitting and improve performance
		y = F.dropout(y,p=0.9)
		#Produce two-class classification probabilities
		y = self.softmax(y)
		return y

	def initHidden(self,mini_batch_size=79):
		#Return the two variables required to initialize the Neural Network
		#(here initialized to zero)
		if torch.cuda.is_available():
			return(Variable(torch.randn(self.nb_layer,mini_batch_size, self.hidden_size).cuda()),
				Variable(torch.randn(self.nb_layer,mini_batch_size, self.hidden_size)).cuda())
		else:
			return(Variable(torch.randn(self.nb_layer,mini_batch_size, self.hidden_size)),
				Variable(torch.randn(self.nb_layer,mini_batch_size, self.hidden_size)))
		
###################
#  TRAIN FUNCTION #
###################

def train(model,train_target, train_input,test_target,test_input,mini_batch_size=79):
	#CrossEntropyLoss used because of probabilty output
	criterion = nn.CrossEntropyLoss()

	#Other optimizer ovefits or produce less good results
	#optimizer = optim.SGD(model.parameters(), lr = 0.1)
	optimizer = optim.RMSprop(model.parameters(), lr = 1e-4)
	#optimizer = optim.Adam(model.parameters(), lr = 1e-3)

	#ploting list initilization
	total_loss=[]
	error_plot_train=[]
	error_plot_test=[]

	#number of epoch set to 75, enough to visualize convergence
	nb_epochs = 150

	#lambda parameter for L1/L2 regularization try
	lamb2=0.002
	lambda_1=0.00005
	#Epoch loop
	for e in range(0, nb_epochs):
		#batch loop
		for b in range(0, train_input.size(0), mini_batch_size):
			#initialisation of the model
			model.hidden = model.initHidden(mini_batch_size)
			model.zero_grad()

			#Output and loss computation
			output=model(train_input.narrow(0, b, mini_batch_size))
			loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
			
			##L2 regularization
			for p in  model.parameters ():
				loss +=  lamb2 * p.pow (2).sum()

			#gradient backpropagation and parameter update
			loss.backward()
			#Clip grad to prevent overfitting and improve results
			torch.nn.utils.clip_grad_norm(model.parameters(),0.25)
			optimizer.step()

			##L1 regularization
			for p in  model.parameters ():
				p.data  -= p.data.sign() * p.data.abs().clamp(max = lambda_1)	
		
		#plotting computation
		total_loss.append(loss.data[0])
		error_plot_train.append(compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100)
		
		#model set to test because of dropout
		model.train(False)
		error_plot_test.append(compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100)
		model.train(True)
		print('training {:.02f}% loss  {:.05f} train error {:.02f}% test error {:.02f}%'.format((e/nb_epochs)*100,
			total_loss[-1],
			error_plot_train[-1],
			error_plot_test[-1]))
	return total_loss,error_plot_train,error_plot_test


###############
#  TEST ERORS #
###############

def compute_nb_errors(model, test_input, test_target,mini_batch_size=79):

	nb_data_errors = 0
	#size modification to adapt batch size to test size
	if test_input.size()[0] == 100:
		mini_batch_size=10
	#on batch loop
	for b in range(0, test_input.size(0), mini_batch_size):
		#initialization
		model.hidden = model.initHidden(mini_batch_size)
		output = model(test_input.narrow(0, b, mini_batch_size))
		#predicted class taken from the output
		_, predicted = torch.max(output.data, 1)
		#numbers of errors computation
		for k in range(0, mini_batch_size):
			if test_target.data[b + k] != predicted[k]:
				nb_data_errors = nb_data_errors + 1
	return nb_data_errors

############
#  MAIN    #
############

if __name__=='__main__':

	########
	# DATA #
	########

	train_input , train_target = bci.load(root =  './data_bci',one_khz = False)
	print(str(type(train_input)), train_input.size())
	print(str(type(train_target)), train_target.size())
	print(train_input[5,:,])
	test_input , test_target = bci.load(root =  './data_bci', train = False,one_khz = False)
	print(str(type(test_input)), test_input.size())
	print(str(type(test_target)), test_target.size())

	#normalize the train and test data (improves the performance)
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)
	test_input.sub_(mean).div_(std)

	#Variable definition with cuda
	#remove .cuda() if not available
	train_input, train_target = Variable(train_input).cuda(), Variable(train_target).cuda()
	test_input, test_target = Variable(test_input).cuda(), Variable(test_target).cuda()

	#model parameter definition
	mini_batch_size = 79
	n_hidden = 65
	n_input = train_input.size()[1]
	n_output = 2

	#model instantiation
	#remove the cuda line if not available
	lstm = LSTM(n_input, n_hidden, n_output)
	lstm.cuda()

	#training and plot
	total_loss,error_plot_train,error_plot_test=train(lstm,train_target,train_input,test_target,test_input,mini_batch_size)
	plt.figure()
	plt.plot(error_plot_train, label='train error')
	plt.plot(error_plot_test, label='test error')
	plt.legend()
	plt.show()