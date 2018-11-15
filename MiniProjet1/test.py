from MiniProjet1_LSTM import *
import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dlc_bci as bci

if __name__=='__main__':

	####################
	# DATA PREPARATION #
	####################

	train_input , train_target = bci.load(root =  './data_bci',one_khz = False)
	test_input , test_target = bci.load(root =  './data_bci', train = False,one_khz = False)

	#normalize the train and test data (improves the performance)
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)
	test_input.sub_(mean).div_(std)

	#Variable definition with cuda
	#remove .cuda() if not available
	if torch.cuda.is_available():
		print("Running cuda version")
		train_input, train_target = Variable(train_input).cuda(), Variable(train_target).cuda()
		test_input, test_target = Variable(test_input).cuda(), Variable(test_target).cuda()
	else:
		print("Running non cuda version")
		train_input, train_target = Variable(train_input), Variable(train_target)
		test_input, test_target = Variable(test_input), Variable(test_target)
	
	#model parameter definition
	mini_batch_size = 79
	n_hidden = 65
	n_input = train_input.size()[1]
	n_output = 2

	#plot end mean
	fig, axes = plt.subplots(2, 5, figsize=(15, 5))
	perf=[]
	for i,ax in enumerate(axes.ravel()):
		#model instantiation
		#remove the cuda line if not available
		lstm = LSTM(n_input, n_hidden, n_output)
		if torch.cuda.is_available():
			lstm.cuda()

		#training and plot
		total_loss,error_plot_train,error_plot_test=train(lstm,train_target,train_input,test_target,test_input,mini_batch_size)
		ax.plot(error_plot_train, label='train error')
		ax.plot(error_plot_test, label='test error')
		ax.legend()
		ax.set_xlabel('epoch')
		ax.set_ylabel('error rate (%)')
		perf.append(min(error_plot_test))
	print('Average performance: {:.02f} %'.format(np.mean(perf)))
plt.show()