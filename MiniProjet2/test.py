#-*-coding:utf8 -*
#############################
#       MINI Project 2      #
#         test file         #
#                           #
# Version 6                 #
#############################
import model
from model import Linear, TanhS, ReLu, MSELoss, Sequential, Module, SGD, Dropout
from torch import FloatTensor
from torch import LongTensor
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch import Tensor

import random

#######################################
# Class Custom_Model                  #
# custom model defined for testing    #
#######################################
class Custom_Model(Module):
	def __init__(self):
        #contructor
        #model inherits from the module class
		Module.__init__(self)
        #the model is constitued of 2 linear layer with activation layer
		self.l1 = Sequential(Linear(2, 16), ReLu(), Linear(16, 92))
		self.s1 = TanhS()
		self.l2 = Linear(92, 2)
	
	def forward(self, input):
        #forward pass defined as in pytorch
		input = self.l1.forward(input)
		output = self.l2.forward(self.s1.forward(input))
		return output

	def backward(self, dlp):
        #backward pass is only backward on all layers
		dlp = self.l2.backward(dlp)
		dlp = self.s1.backward(dlp)
		dlp = self.l1.backward(dlp)
		return dlp

#######################################
# Function create_deep_model          #
# create a MLP with Pytorch           #
#######################################
def create_deep_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

#######################################
# Function generate_disc_set          #
# Generate the toy dataset            #
# taken from practical sessions       #
#######################################
def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    test_target = Tensor(nb, 2).fill_(0.0)
    #Not tested
    target = input.sub(0.5).pow(2).sum(1).sub(1.0 / (2.0 * math.pi)).sign().sub(1).div(2).abs().long()
    test_target[:, 1] = target[:]
    test_target[:, 0] = target.sub(1).abs()[:]
    return input, test_target, target

# Generating data. train_input -> (x, y)
#		   train_target -> (1, 0) if (x, y) not in the 1/sqrt(2pi) circle else (0, 1). For MSELoss
#		   target -> 0 if (x, y) not in the 1/sqrt(2pi) circle else 1. For CrossEntropy (or human ;)
def generate_data(n):
	train_input, train_target, train_label = generate_disc_set(n)
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)

	return train_input, train_target, train_label

# Train a torch model 
def train_torch_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr = 0.05)
    for e in range(0, 250):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.data[0]
            opt.step()
        print(e, sum_loss)

# train a custom model
def train_model(model, train_input, train_target, mini_batch_size):
    criterion = MSELoss(model)
    opt = SGD(model, lr=0.05)
    model.fit(train_input, train_target, opt, criterion, mini_batch_size, epoch=250)

# compute nb of errors on a torch model
def compute_torch_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# compute nb of errors on a custom model
def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(0, mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


if __name__ == "__main__":

	# The seed is set to 0 during the part of the models definitions. Weights of the linear layers are randomly 
	# initialized. Sometimes, these values doesn't make the model converge (whether it is our or pytorch one, as it is the same architecture). 
	# In real case, the user just have to relaunch the process but as it is for evaluation purpose, we
	# prefer our script to have a predictable result. 
	# To put it on a nutshell, we initialize the weights with deterministic values.
	torch.manual_seed(0) 	

	# Model definitions
	model = Sequential(Linear(2, 25), ReLu(), Linear(25,25), ReLu(),  Linear(25,25), ReLu(), Dropout(0.2), Linear(25, 2), ReLu())
	model_torch = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25,25), nn.ReLU(), nn.Linear(25,25), nn.ReLU(),nn.Dropout(0.2), nn.Linear(25, 2), nn.ReLU())
	# Creating toy datas

	# Set the seed to a random value, this time to generate data randomly (and for the dropout layers)
	torch.manual_seed(random.randint(0,2**32 - 1))
	
	train_input, train_target, label = generate_data(10000)
	test_input, test_target, test_label = generate_data(200)
	
	# Training models 
	train_model(model, train_input, train_target, 500)
	model_torch.train()
	train_torch_model(model_torch,  Variable(train_input), Variable(train_target), 500)
	model_torch.eval()

	# Computing errors
	print("custom model error {:.2f}%".format(compute_nb_errors(model, test_input, test_label, 20) / 2.0))
	print("torch model error {:.2f}%".format(compute_torch_nb_errors(model_torch, Variable(test_input), Variable(test_label), 20) / 2.0))
