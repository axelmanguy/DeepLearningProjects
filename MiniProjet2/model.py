#-*-coding:utf8 -*
#############################
#       MINI Project 2      #
#         framework         #
#                           #
# Version 6                 #
#############################

from collections import OrderedDict
from torch import FloatTensor
from torch import LongTensor
import math
import sys

#######################################
# Class Parameter                     #
# Defines a model parameter           #
# Also store the associated gradients #
#######################################

class Parameter:
	#constructor
	def __init__(self, data):
		#data contains the parameter itself
		self.data = data
		#data contains the gradient
		#associated with that parameter
		self.grad = data.new(data.size())
		#Gradient instantiated to 0
		self.grad.fill_(0.0)

#########################################
# Class Module                          #
# Abstract class implementing shared    #
# methods and variables for all modules #
#########################################

class Module(object):
	#Constructor
	def __init__(self):
		#non public _modul stores all the submodules in order
		self._module = OrderedDict()
		#parameter list store all the parameter of the current Module
		self.parameter = []
		self.train = False


	def setTraining(self, v):
		#recursive method to set training mode on all submodules
		self.train = v		
		for m in self._module.values():
			m.setTraining(v)

	def forward(self, input):
		#Abstract method have to be implemented by descendants
		raise NotImplementedError

	def backward(self):
		#Abstract method have to be implemented by descendants
		raise NotImplementedError

	def parameters(self):
		#parameter method define a generator for the parameter
		for param in self.parameter:
			#return an iterator for parameters
			yield param
		for m in self._module.values():
			for param in m.parameters():
				#return an iterator for each parameters
				#for all submodules
				yield param

	def zero_grad(self):
		#zero grad method initializa all gardient to zero
		#for current module (and submodules)
		for param in self.parameter:
			param.grad.fill_(0.0)	
		for m in self._module.values():
			m.zero_grad()

	def __setattr__(self, name, value):		
		#setter method for adding modules
		# When a new attribute is defined, 
		# store it in the module dictinnary.
		module = self.__dict__.get("_module", None)
		#check if it's Module
		if isinstance(value, Module):
			module[name] = value
		#overwrite if already existing
		elif module is not None and name in module:
			del module[name]
		#set the attribute
		object.__setattr__(self, name, value)

	def apply(self, f):
		#apply method for applying a specific function to
		# all the submodules
		f(self)
		for m in self._module.values():
			m.apply(f)
	
	def _fancyBar(self, e, Ne, loss):
		#private method for graphical display
		#write output in a Kera inspired way
		if e == 0:
			print("")
		
		n = int((float(e)/float((Ne-1)))*18)
		sys.stdout.write("\rEpoch {:2d}/{:2d} [".format(e, Ne)+"="*n+" "*(18-n)+"]"+" loss:{}".format(loss))
		sys.stdout.flush()
		if e == (Ne - 1):
			print("")

	def fit(self, train_input, train_target, opt, criterion, mini_batch_size, epoch=1):
		#convenient method implementing the traingin in the modules
		#set trainingmode
		self.setTraining(True)
		#loop on epoch
		for e in range(0, epoch):
			sum_loss = 0
			#loop on batch
			for b in range(0, train_input.size(0), mini_batch_size):
				batch_size = mini_batch_size if train_input.size(0) - b > mini_batch_size else train_input.size(0) - b
				loss = criterion.forward(train_input.narrow(0, b, batch_size), train_target.narrow(0, b, batch_size))
				self.zero_grad()
				criterion.backward()
 				sum_loss = sum_loss + loss
				opt.step()
			#bar display implementation
			self._fancyBar(e, epoch, sum_loss)
			#print("Total loss of this epoch : {}".format(sum_loss))
		self.setTraining(False)
		
##############################################
# Class Dropout                              #
# Implements the dropout layer               #
# remove input according to a p probability  #
##############################################
class Dropout(Module):
	#constructor p is 0.5 by default
	def __init__(self, p=0.5):
		Module.__init__(self)
		self.p = p
		self.bernoulli_matrix = FloatTensor()
	
	def forward(self, input):
		#if model in train mode
		if self.p > 0 and self.train:
			self.bernoulli_matrix.resize_as_(input)
			self.bernoulli_matrix.bernoulli_(1.0 - self.p)
			self.bernoulli_matrix.div_(1.0 - self.p)
		#if model in test mode
		else:
			self.bernoulli_matrix.resize_as_(input)
			self.bernoulli_matrix.fill_(1.0)	

		output = input.mul(self.bernoulli_matrix)
		return output

	def backward(self, dlp):
		#backward pass; dlp is the previous layer derivative
		dl = dlp
		if self.p > 0 and self.train:
		 	dl = dlp * self.bernoulli_matrix.t()
		return dl
		
#########################################
# Class MSELoss                         #
# Implements the loss computation       #
# Using Mean Square Error               #
#########################################
class MSELoss:
	def __init__(self, model):
		#constructor, MSELoss instantiated for a single model
		self.model = model

	def forward(self, input, target):
		#forward pass, first compute next output
		output = self.model.forward(input)
		self.last_output = output
		self.last_target = target
		#compute the mean square error between output and target
		return (output - target).pow(2).mean()

	def backward(self):
		#backward pass
		sample_last_output = self.last_output
		sample_last_target = self.last_target
		#derived loss
		dl_dx = 2.0 * (sample_last_output - sample_last_target) / sample_last_output.size(1)
		#update the model backward and return
		self.model.backward(dl_dx.t())

#########################################
# Class TanhS                           #
# Implements the TanhS activation layer #
#########################################
class TanhS(Module):
	def __init__(self):
		#Constructor
		#inherits from Module class
		Module.__init__(self)

	def forward(self, input):
		#forward pass method
		self.last_input = input
		#apply tanh on input
		output = input.tanh()
		#return result
		return output

	def _dsigma(self, x):
		#return derived tanh
		return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

	def backward(self, dlp):
		#backward pass; dlp is the previous layer derivative 
		#return the gradient of activation layer
		#times the derivative of the previous layer
		dl_ds = self._dsigma(self.last_input.t()) * dlp
		return dl_ds

#########################################
# Class ReLu                            #
# Implements the ReLu activation layer  #
#########################################	
class ReLu(Module):
	def __init__(self):
		#Constructor
		#inherits from Module class
		Module.__init__(self)

	def forward(self, input):
		#forward pass method
		self.last_input = input
		#apply Relu on input
		output = (input + input.abs()).div(2.0)
		#return result
		return output

	def backward(self, dlp):
		#backward pass; dlp is the previous layer derivative
		dl_ds = self.last_input.t().sign().add(1).div(2.0) * dlp
		return dl_ds
	
#########################################
# Class Linear                          #
# Implements a linear layer             #
#########################################
class Linear(Module):
	
	def __init__(self, in_features, out_features, bias=True):
		#Constructor
		#inherits from Module class
		Module.__init__(self)

		# Attributes
		self.in_features = in_features
		self.out_features = out_features
		

		# Linear layer parameters
		# adding weights to parameter attribute
		self.weight = Parameter(FloatTensor(out_features, in_features))
		self.parameter.append(self.weight)		
		# adding bias to parameter attribute
		if bias:
			self.bias = Parameter(FloatTensor(out_features))
			self.parameter.append(self.bias)
		else:
			self.bias = None

		# Standard initialization
		self.reset_parameters()

	
	def forward(self, input):		
		#Forward pass
		self.last_input = input
		#multiply all entry vector by weights
		output = input.matmul(self.weight.data.t())
		#add the bias to the output
		if self.bias is not None:
			output += self.bias.data
		#return output
		self.last_output = output
		return output

	def backward(self, dlp):		
		#backward pass
		# compute derivative of loss
		dl_dw = dlp.mm(self.last_input)
		#add the gradient to the parameter
		self.weight.grad.add_(dl_dw).div_(self.last_input.size(0))
		if self.bias is not None:
			self.bias.grad.add_(dlp.mean(dim=1)).div_(self.last_input.size(0))
		#for i in range(self.last_input.size(0)):
		#	sample_last_input = self.last_input.narrow(0, i, 1)
		#	sample_dlp = dlp.narrow(1, i, 1)
		#	dl_dw = sample_dlp.mm(sample_last_input)
		#	self.weight.grad.add_(dl_dw)
		#	if self.bias is not None:
		#		self.bias.grad.add_(sample_dlp.squeeze())
		#self.weight.grad.div_(self.last_input.size(0))
		#if self.bias is not None:
		#	self.bias.grad.div_(self.last_input.size(0))

		#return the layer gradient 
		dl_dx = self.weight.data.t().mm(dlp)
		return dl_dx

	def reset_parameters(self):
		#reset parameters method from pytorch
		stdv = 1. / math.sqrt(self.weight.data.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)


#########################################
# Class Sequential                      #
# Implements a sequence of modules      #
#########################################
class Sequential(Module):
	def __init__(self, *layer):
		#Constructor
		#inherits from Module class
		Module.__init__(self)

		# Update attributes
		self.layer = [l for l in layer]

		#Set all the submodules
		for i, l in enumerate(self.layer):
			Module.__setattr__(self, "layer_{}".format(i), l)

	def forward(self, input):
		#forward pass, apply forward pass on all submodules
		for l in self.layer:
			input = l.forward(input)
		return input

	def backward(self, dlp):
		#backward pass, apply backward pass on all submodules
		for l in reversed(self.layer):
			dlp = l.backward(dlp)
		return dlp

#########################################
# Class Optimizer                       #
# Abstract method in cass we want to    #
# build other optimizer                 #
#########################################
class Optimizer:

	def __init__(self, model):
		#simple constructor
		self.model = model
	
	def step(self):
		#Abstract method have to be implemented by descendants
		raise NotImplementedError

#########################################
# Class SGD                             #
# Implements the stochastic gradient    #
# descent refer to report for formula   #
#########################################
class SGD(Optimizer):
	def __init__(self, model, lr, momentum = 0.0, decay = 0.0):
		Optimizer.__init__(self, model)
		self.previous_value = [0 for p in model.parameters()]
		self.momentum = momentum
		self.decay = decay
		self.lr = lr

	def step(self):
		value = []
		#iteration on all parameter
		for i, p in enumerate(self.model.parameters()):
			v = (self.momentum *  self.previous_value[i] + (1.0 - self.momentum) * p.grad)
			value.append(v)
			p.data.sub_(self.lr * v)

		self.lr = self.lr - self.decay if self.lr - self.decay > 0 else 0

		self.previous_value = value 
            		
	

#########################################
# Function init_weight                  #
# Just used to init the weight and the  #
# bias of each linear level to the      #
# same value                            #
#########################################
def init_weights(m):
	if type(m) == nn.Linear or isinstance(m, Linear):
		m.weight.data.fill_(0.5)
		m.bias.data.fill_(0.5)

if __name__ == "__main__":
	"This package as no meaning being executed, it only contains our model implementation"
