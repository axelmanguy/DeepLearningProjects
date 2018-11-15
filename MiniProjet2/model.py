#-*-coding:utf8 -*
from collections import OrderedDict
from torch import FloatTensor
from torch import LongTensor
import math
import sys

# Embedded a data with its gradient
class Parameter:
	def __init__(self, data):
		self.data = data
		self.grad = data.new(data.size())
		self.grad.fill_(0.0)
# Mother class
class Module(object):
	def __init__(self):
		self._module = OrderedDict()
		self.parameter = []
		self.train = False


	def setTraining(self, v):
		self.train = v		
		for m in self._module.values():
			m.setTraining(v)

	def forward(self, input):
		raise NotImplementedError

	def parameters(self):
		for param in self.parameter:
			yield param
		for m in self._module.values():
			for param in m.parameters():
				yield param


	def zero_grad(self):
		for param in self.parameter:
			param.grad.fill_(0.0)	
		for m in self._module.values():
			m.zero_grad()

	# When a attribute is create, like self.l1 = Linear(2, 4), store it in the module dictinnary.
	def __setattr__(self, name, value):
		module = self.__dict__.get("_module", None)
		if isinstance(value, Module):
			module[name] = value
		elif module is not None and name in module:
			del module[name]
		object.__setattr__(self, name, value)

	def backward(self):
		raise NotImplementedError

	def apply(self, f):
		f(self)
		for m in self._module.values():
			m.apply(f)
	
	def _fancyBar(self, e, Ne, loss):
		if e == 0:
			print("")
		
		n = int((float(e)/float((Ne-1)))*18)
		sys.stdout.write("\rEpoch {:2d}/{:2d} [".format(e, Ne)+"="*n+" "*(18-n)+"]"+" loss:{}".format(loss))
		sys.stdout.flush()
		if e == (Ne - 1):
			print("")

	def fit(self, train_input, train_target, opt, criterion, mini_batch_size, epoch=1):

		self.setTraining(True)
		for e in range(0, epoch):
			sum_loss = 0
			for b in range(0, train_input.size(0), mini_batch_size):
				batch_size = mini_batch_size if train_input.size(0) - b > mini_batch_size else train_input.size(0) - b
				loss = criterion.forward(train_input.narrow(0, b, batch_size), train_target.narrow(0, b, batch_size))
				self.zero_grad()
				criterion.backward()
				sum_loss = sum_loss + loss
				opt.step()
			self._fancyBar(e, epoch, sum_loss)
			#print("Total loss of this epoch : {}".format(sum_loss))
		self.setTraining(False)
		


class Dropout(Module):
	def __init__(self, p=0.5):
		Module.__init__(self)
		self.p = p
		self.bernoulli_matrix = FloatTensor()
	
	def forward(self, input):
		if self.p > 0 and self.train:
			self.bernoulli_matrix.resize_as_(input)
			self.bernoulli_matrix.bernoulli_(1.0 - self.p)
			self.bernoulli_matrix.div_(1.0 - self.p)

		else:
			self.bernoulli_matrix.resize_as_(input)
			self.bernoulli_matrix.fill_(1.0)	

		output = input.mul(self.bernoulli_matrix)
		return output

	def backward(self, dlp):
		dl = dlp
		if self.p > 0 and self.train:
		 	dl = dlp * self.bernoulli_matrix.t()
		return dl
		
		


# Mean Square Error loss function
class MSELoss:
	def __init__(self, model):
		self.model = model

	def forward(self, input, target):
		output = self.model.forward(input)
		self.last_output = output
		self.last_target = target
		return (output - target).pow(2).mean()

	def backward(self):
		sample_last_output = self.last_output
		sample_last_target = self.last_target
		dl_dx = 2.0 * (sample_last_output - sample_last_target) / sample_last_output.size(1)
		self.model.backward(dl_dx.t())

# Activation function
class TanhS(Module):
	def __init__(self):
		Module.__init__(self)

	def forward(self, input):
		self.last_input = input
		output = input.tanh()
		return output

	def _dsigma(self, x):
		return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

	def backward(self, dlp):
		dl_ds = self._dsigma(self.last_input.t()) * dlp
		return dl_ds
	
class ReLu(Module):
	def __init__(self):
		Module.__init__(self)

	def forward(self, input):
		self.last_input = input
		output = (input + input.abs()).div(2.0)
		return output

	def backward(self, dlp):
		dl_ds = self.last_input.t().sign().add(1).div(2.0) * dlp
		return dl_ds
	

# Linear Layer
class Linear(Module):
	
	def __init__(self, in_features, out_features, bias=True):
		Module.__init__(self)

		# Attributes
		self.in_features = in_features
		self.out_features = out_features
		

		# Parameters
		self.weight = Parameter(FloatTensor(out_features, in_features))
		self.parameter.append(self.weight)
		if bias:
			self.bias = Parameter(FloatTensor(out_features))
			self.parameter.append(self.bias)
		else:
			self.bias = None

		# Standard initialization
		self.reset_parameters()

	
	def forward(self, input):
		self.last_input = input
		output = input.matmul(self.weight.data.t())
		if self.bias is not None:
			output += self.bias.data

		return output

	def backward(self, dlp):
		dl_dw = dlp.mm(self.last_input)
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


		dl_dx = self.weight.data.t().mm(dlp)
		return dl_dx



	# Source: pytorch
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.data.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)



class Sequential(Module):
	def __init__(self, *layer):
		Module.__init__(self)

		# Attributes
		self.layer = [l for l in layer]

		#Modules
		for i, l in enumerate(self.layer):
			Module.__setattr__(self, "layer_{}".format(i), l)

	def forward(self, input):
		for l in self.layer:
			input = l.forward(input)
		return input

	def backward(self, dlp):
		for l in reversed(self.layer):
			dlp = l.backward(dlp)
		return dlp

class Optimizer:

	def __init__(self, model):
		self.model = model
	
	def step(self):
		raise NotImplementedError

class SGD(Optimizer):
	def __init__(self, model, lr, momentum = 0.0, decay = 0.0):
		Optimizer.__init__(self, model)
		self.previous_value = [0 for p in model.parameters()]
		self.momentum = momentum
		self.decay = decay
		self.lr = lr

	def step(self):
		value = []
		
		for i, p in enumerate(self.model.parameters()):
			v = (self.momentum *  self.previous_value[i] + (1.0 - self.momentum) * p.grad)
			value.append(v)
			p.data.sub_(self.lr * v)

		self.lr = self.lr - self.decay if self.lr - self.decay > 0 else 0

		self.previous_value = value 
            		
	

# Just use to init the weight and the bias of each linear level to the same value
def init_weights(m):
	if type(m) == nn.Linear or isinstance(m, Linear):
		m.weight.data.fill_(0.5)
		m.bias.data.fill_(0.5)


# Use to test if torch and custom implementation give the same results
if __name__ == "__main__":
	"This package as no meaning being executed, it only contains our model implementation"
