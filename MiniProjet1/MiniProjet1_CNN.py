# Import modules
import torch
from torch import Tensor
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Loading Data
import dlc_bci as bci
train_input,train_target=bci.load(root='.\data_bci',one_khz=False)
print(str(type(train_input)),train_input.size())
print(str(type(train_target)),train_target.size())
test_input,test_target=bci.load(root='.\data_bci',train=False,one_khz=False)
print(str(type(test_input)),test_input.size())
print(str(type(test_input)),test_target.size())

# Because self.conv2d requires a 4D tensor, with number of channels specified in the 
# 2nd position, the data needs to be re-structured
train_input=torch.unsqueeze(train_input,1)
print(str(type(train_input)),train_input.size())
test_input=torch.unsqueeze(test_input,1)
print(str(type(test_input)),test_input.size())

# Normalisation
mean,std=train_input.mean(),train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)
train_input,train_target=Variable(train_input),Variable(train_target)
test_input,test_target=Variable(test_input),Variable(test_target)

#Model Definition : CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,50, kernel_size=5)
        self.conv2 = nn.Conv2d(50,10, kernel_size=5)
        self.drop2D = nn.Dropout2d(p=0.90, inplace=False)
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.leaky_relu(self.fc1(x.view(-1, 100)))
        x = self.fc2(x)
        return x
    
#For Training Model         
def train(model,train_target, train_input,mini_batch_size=79):
    total_loss=[]
    error_plot_train=[]
    error_plot_test=[]
    #Number of iterations
    nb_epochs = 400
    #Optimiser
    optimiser = optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    
    for e in range(0, nb_epochs):
        print('training : {:.02f}%'.format((e/nb_epochs)*100))
        for b in range(0, train_input.size(0), mini_batch_size):
            optimiser.zero_grad()
            output=model(train_input.narrow(0,b,mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss.backward()
            optimiser.step()
            
            total_loss.append(loss.data[0])
        error_plot_train.append(compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100)
        error_plot_test.append(compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100)
    return total_loss,error_plot_train,error_plot_test

#To Compute Total Number of Errors
def compute_nb_errors(model, test_input, test_target,mini_batch_size=79):
    nb_data_errors = 0
    if test_input.size()[0] == 100:
        mini_batch_size= 10
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input.narrow(0, b, mini_batch_size))
        _, predicted = torch.max(output.data, 1)
        for k in range(0, mini_batch_size):
            if test_target.data[b + k] != predicted[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors

#Code segment for running all functions
mini_batch_size = 79
n_input = train_input.size()[1]
n_output = 2
net = Net()
total_loss,error_plot_train,error_plot_test=train(net,train_target, train_input,mini_batch_size=79)
print('train_error {:.02f}%  '.format(compute_nb_errors(net, train_input, train_target) / train_input.size(0) * 100))
print('test_error {:.02f}%  '.format(compute_nb_errors(net, test_input, test_target) / test_input.size(0) * 100))
plt.figure()
plt.plot(error_plot_train, label='train error')
plt.plot(error_plot_test, label='test error')
plt.legend()
plt.show()