#############################
#       MINI Project 1      #
#         rnn model         #
# Version 3                 #
# Date 22/04/2018           #
# Author: manguy            #
#############################

#######################
# LIBRARY IMPORTATION #
#######################

import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import dlc_bci as bci

####################
# DATA PREPARATION #
####################

train_input , train_target = bci.load(root =  './data_bci',one_khz = False)
#print(str(type(train_input)), train_input.size())
#print(str(type(train_target)), train_target.size())
#print(train_input[5,:,])
test_input , test_target = bci.load(root =  './data_bci', train = False,one_khz = False)
#print(str(type(test_input)), test_input.size())
#print(str(type(test_target)), test_target.size())

#normalize the train and test data (improves the performance)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#Variable definition with cuda
#remove .cuda() if not available
train_input, train_target = Variable(train_input).cuda(), Variable(train_target).cuda()
test_input, test_target = Variable(test_input).cuda(), Variable(test_target).cuda()

#####################
#  MODEL DEFINITION #
#####################

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		#Definition of the hidden_size
		self.hidden_size = hidden_size

		#first layer from input to hidden
		self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)

		#second layer from hidden to output (size = 2)
		self.input2ouput = nn.Linear(input_size + hidden_size, output_size)

		#LogSoftmax to produce two-class classification probabilities
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):

		#first concatenate the previous output from hidden with the new slice
		#of the input (next frame)
		combined = torch.cat((input, hidden), 1)

		#Then pass into hidden layer( will be concatenate at next forward call)
		hidden = self.input2hidden(combined)

		#From hidden to 2 class output
		output = self.input2ouput(combined)

		#dropout to improve performance
		#not produce better results
		#output = F.dropout(output,p=0.1)

		output = self.softmax(output)
		return output, hidden

	def initHidden(self,batch_size):
		#network needs to be initialized for the first value of sequence
		#(here initialized to zero)
		return Variable(torch.zeros(batch_size, self.hidden_size)).cuda()


###################
#  TRAIN FUNCTION #
###################

def train(model,train_target, train_input,mini_batch_size=79):
	#CrossEntropyLoss used because of probabilty output
	criterion = nn.CrossEntropyLoss()

	#Other optimizer ofen tends to overfit
	optimizer = optim.SGD(model.parameters(), lr = 1e-1)

	#ploting list initilization
	total_loss=[]
	error_plot_train=[]
	error_plot_test=[]
	#number of epoch set to 75, enough to visualize convergence
	nb_epochs = 75
	#Epoch loop
	for e in range(0, nb_epochs):
		#batch loop
		for b in range(0, train_input.size(0), mini_batch_size):
			#initialisation of Rnn
			hidden = rnn.initHidden(mini_batch_size)
			model.zero_grad()
			#sequential loop the output is the final output
			#(At the end of the sequence)
			for i in range(train_input.narrow(0, b, mini_batch_size).size()[2]):
				output, hidden = model(train_input.narrow(0, b, mini_batch_size)[:,:,i],hidden)
			loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
			#backward computation
			loss.backward()

			#Clip grad to prevent overfitting and improve results
			torch.nn.utils.clip_grad_norm(model.parameters(),0.25)

			#next step
			optimizer.step()

		#plotting computation
		total_loss.append(loss.data[0])
		error_plot_train.append(compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100)
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
		#initialisation
		hidden = model.initHidden(mini_batch_size)
		#sequence loop
		for i in range(train_input.narrow(0, b, mini_batch_size).size()[2]):
			output,hidden = model(test_input.narrow(0, b, mini_batch_size)[:,:,i],hidden)
		#predicted classes taken from the last output
		#the one that correponds to the whole sequence
		_, predicted_classes = torch.max(output.data, 1)
		#numbers of errors computation
		for k in range(0, mini_batch_size):
			if test_target.data[b + k] != predicted_classes[k]:
				nb_data_errors = nb_data_errors + 1
	return nb_data_errors

############
#  MAIN    #
############

if __name__=='__main__':
	#model parameter definition
	#f, axes = plt.subplots(2, 2)
	#for i,ax in enumerate(axes.ravel()):
	mini_batch_size = 79
	n_hidden = 128
	n_input = train_input.size()[1]
	n_output = 2

	#model instantiation
	#remove the cuda line if not available
	rnn = RNN(n_input, n_hidden, n_output)
	rnn=rnn.cuda()

	#training
	total_loss,error_plot_train,error_plot_test=train(rnn,train_target,train_input,mini_batch_size)
	
	#plot the results
	plt.plot(error_plot_train, label='train error')
	plt.plot(error_plot_test, label='test error')
	plt.xlabel('epoch')
	plt.ylabel('error in %')
	plt.legend()
	plt.show()