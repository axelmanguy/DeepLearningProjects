import matplotlib.pyplot as plt

import dlc_bci as bci

import torch
import math
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn

MINI_BATCH_SIZE = 50


def classToResult(result):
	_, i = torch.max(result, 1)
	return i

def getTrainData():
	train_input, train_target = bci.load(root = './data_bci')
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)
	train_input, train_target = Variable(train_input, requires_grad=True), Variable(train_target)
	return train_input, train_target


def getTestData():
	test_input, test_target = bci.load(root = './data_bci', train = False)
	mean, std = test_input.mean(), test_input.std()
	test_input.sub_(mean).div_(std)
	test_input, test_target = Variable(test_input), Variable(test_target)
	return test_input, test_target


def computeNbErrors(model, data_input, data_target):
	mini_batch_size = MINI_BATCH_SIZE
	nb_data_errors = 0

	for b in range(0, data_input.size(0), mini_batch_size):
		current_batch_size = mini_batch_size if b + mini_batch_size <= data_input.size(0) else data_input.size(0) - b
		output = model(data_input.narrow(0, b, mini_batch_size).view(current_batch_size, 1400))
		_, predicted_classes = torch.max(output.data, 1)
		for k in range(0, mini_batch_size):
			if data_target.data[b + k] != predicted_classes[k]:
				nb_data_errors = nb_data_errors + 1

	return nb_data_errors


def trainModelWithError(model, train_input, train_target, test_input, test_target):
	l_e = []
	mini_batch_size = MINI_BATCH_SIZE
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum=0.003)
	nb_epochs = 1000

	for e in range(0, nb_epochs):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			current_batch_size = mini_batch_size if b + mini_batch_size < train_input.size(0) else train_input.size(0) - b
			output = model(train_input.narrow(0, b, current_batch_size).view(current_batch_size, 1400))
			loss = criterion(output, train_target.narrow(0, b, current_batch_size))
			model.zero_grad()
			loss.backward()
			optimizer.step()
			sum_loss += loss.data[0]
		nb_e = computeNbErrors(model, test_input, test_target)
		print(nb_e, sum_loss)
		l_e.append(nb_e)
	return l_e



def createDeepModel():
    return nn.Sequential(nn.Linear(1400, 8), nn.ReLU(), nn.Linear(8, 2), nn.ReLU())



def main():
	train_input, train_target = getTrainData()
	test_input, test_target = getTestData()
	
	model = createDeepModel()
	l_e = trainModelWithError(model, train_input, train_target, test_input, test_target)
	nb_errors = computeNbErrors(model, test_input, test_target)
	print("The model got {:.1f}% errors".format((float(nb_errors) / test_input.size()[0]) * 100))
	plt.plot(l_e)
	plt.show()
	




if __name__ == "__main__":
	main()






