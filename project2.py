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
Applies a linear transformation to the incoming data: y = xA^T + b
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
        self.weight = torch.distributions.uniform.Uniform(torch.tensor([-init_bound]), torch.tensor([init_bound])).sample(torch.Size([out_features, in_features]))
        if bias:
            self.bias = torch.distributions.uniform.Uniform(torch.tensor([-init_bound]), torch.tensor([init_bound])).sample(torch.Size([out_features]))
        else:
            self.bias = torch.zeros(out_features)
        self.grad_weight = torch.zeros(self.weight.size())
        self.grad_bias = torch.zeros(self.bias.size())

    def forward (self, input):
        self.input = input
        return self.weight.mv(self.input) + self.bias

    def backward (self, gradwrtoutput):
        self.grad_weight.add_(gradwrtoutput.mm(self.input.t()))
        self.grad_bias.add_(gradwrtoutput.sum(0))
        grad_input = gradwrtoutput.mv(self.weight)
        return grad_input

    def param (self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

class ReLU(Module):

    def forward (self, input):
        zeroes = torch.full(input.size, 0.0, dtype = float )
        return torch.where(input > 0, input, zeroes)
    def backward (self, gradwrtoutput):
        ones = torch.ones(input.size())
        zeroes = torch.full(input.size(), 0.0, dtype = float )
        return torch.where(gradwrtoutput > 0, ones, zeroes)
    def param (self):
        return []

class Tanh(Module):

    def forward (self, input):
        return input.tanh()
    def backward (self, gradwrtoutput):
        return 1 - (gradwrtoutput.tanh()).pow(2)
    def param (self):
        return []

class Sequential(Module):

    def __init__(self, *modules):
        self.modules = list(modules)[0]

    def forward (self, input):
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward (self, gradwrtoutput):
        for module in reversed(self.modules):
            gradwtroutput = module.backward(gradwrtoutput)
        return gradwtroutput

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
        for parameter in self.parameters:
            parameter[0] = parameter[0] - self.eta * parameter[1]


    def zero_grad(self):
        for parameter in self.parameters:
            parameter[1].zero_()
            
