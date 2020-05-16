from torch import empty
import torch
import math

class Module(object):

    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []

'''
Applies a linear transformation to the incoming data
ARGS:
    - weight: the learnable weights
    - bias: the learnable bias
SHAPE:
    - weight: (out_features, in_features)
    - bias: (out_features)

'''
class Linear(Module):

    def __init__(self, in_features, out_features, bias = True):
        init_bound = 1/math.sqrt(in_features)
        self.weight = torch.Tensor(out_features, in_features).uniform_(-init_bound, init_bound)
        if bias:
            self.bias = torch.Tensor(out_features).uniform_(-init_bound, init_bound)
        else:
            self.bias = torch.zeros(out_features)
        self.grad_weight = torch.zeros(self.weight.size())
        self.grad_bias = torch.zeros(self.bias.size())

    def forward (self, input):
        self.previous_layer = input
        return input.mm(self.weight.T) + self.bias

    def backward (self, gradwrtoutput):
        self.grad_weight.add_(gradwrtoutput.T.mm(self.previous_layer))
        self.grad_bias.add_(gradwrtoutput.sum(0))
        grad_input = gradwrtoutput.mm(self.weight)
        return grad_input

    def param (self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

class ReLU(Module):

    def __init__(self):
        self.input = None

    def forward (self, input):
        self.input = input
        zeroes = torch.full(input.size(), 0.0, dtype = float )
        return torch.where(input > 0, input.float(), zeroes.float())
    def backward (self, gradwrtoutput):
        ones = torch.ones(gradwrtoutput.size())
        zeroes = torch.full(gradwrtoutput.size(), 0.0, dtype = float )
        derivative = torch.where(self.input > 0, ones.float(), zeroes.float())
        return gradwrtoutput * derivative
    def param (self):
        return [(None, None)]

class Tanh(Module):

    def __init__(self):
        self.input = None
    def forward (self, input):
        self.input = input
        return input.tanh()
    def backward (self, gradwrtoutput):
        derivative = 1 - (self.input.tanh()).pow(2)
        return gradwrtoutput * derivative
    def param (self):
        return [(None, None)]

class Sequential(Module):

    def __init__(self, *modules):
        self.modules = list(modules)

    def forward (self, input):
        x = input
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward (self, gradwrtoutput):
        x = gradwrtoutput
        for module in reversed(self.modules):
            x = module.backward(x)
        return x

    def param (self):
        parameters = []
        for module in self.modules:
            parameters.append(module.param())
        return parameters

class SGD():

    def __init__(self, parameters, eta):
        self.parameters = parameters
        if eta < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eta))
        else:
            self.eta = eta

    def step(self):
        for module in self.parameters:
            for param, grad_param in module:
                if (param is not None and grad_param is not None):
                    param.sub_(self.eta * grad_param)


    def zero_grad(self):
        for module in self.parameters:
            for param, grad_param in module:
                if (param is not None and grad_param is not None):
                    grad_param.zero_()

class LossMSE():
    def loss(self, prediction, target):
        return (prediction - target).pow(2).sum()

    def grad(self, prediction, target):
        return 2*(prediction - target)

def normalize_data(x):
    mean, std =  x.mean(), x.std()
    x.sub_(mean).div_(std)

def train_and_evaluate(model, train, train_target, test, test_target,
                mini_batch_size, eta, nb_epochs, normalize = True):

    results_list = []

    N_train = train.size(0)
    N_test = test.size(0)

    if normalize:
        normalize_data(train)
        normalize_data(test)


    optimizer = SGD(model.param(), eta)
    MSE = LossMSE()

    for epoch in range(nb_epochs):

        running_accuracy = 0
        train_loss = 0

        for batch in range(0, N_train, mini_batch_size):
            optimizer.zero_grad()
            output = model.forward(train.narrow(0, batch, mini_batch_size))
            train_loss += MSE.loss(output, train_target.narrow(0, batch, mini_batch_size))
            loss_grad = MSE.grad(output, train_target.narrow(0, batch, mini_batch_size))
            _, predicted_classes = output.max(1)
            model.backward(loss_grad)
            optimizer.step()

            running_accuracy += (train_target.narrow(0, batch, mini_batch_size).argmax(1) == predicted_classes).sum().item()

        # compute test loss and accuracy without computing the gradients
        output = model.forward(test)
        test_loss = MSE.loss(output, test_target)
        _, predicted_classes = output.max(1)

        # Compute accuracy
        test_accuracy = (test_target.argmax(1) == predicted_classes).sum().item() / N_test


        results_list.append({"epoch": epoch, "train_loss": train_loss.item() , "test_loss": test_loss.item() , "train_accuracy": running_accuracy/N_train , "test_accuracy": test_accuracy})

    return model, results_list



def generate_data(size):
    input = torch.Tensor(size, 2).uniform_(0, 1)
    target = input.sub(torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target

def one_hot_encoding(target):
    onehot = torch.zeros(target.size(0), 2).fill_(0)
    onehot[range(onehot.shape[0]), target]=1
    return onehot
