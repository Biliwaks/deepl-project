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
        self.weight = torch.Tensor(out_features, in_features).uniform_(-init_bound, init_bound)
        if bias:
            self.bias = torch.Tensor(out_features).uniform_(-init_bound, init_bound)
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
        return [(None, None)]

class Tanh(Module):

    def forward (self, input):
        return input.tanh()
    def backward (self, gradwrtoutput):
        return 1 - (gradwrtoutput.tanh()).pow(2)
    def param (self):
        return [(None, None)]

class Sequential(Module):

    def __init__(self, *modules):
        self.modules = list(modules)

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
        return (target-prediction).pow(2).sum()

    def grad(self, prediction, target):
        return 2*(prediction - target)




def train_model(model, train, train_target, test, test_target,
                mini_batch_size, eta, nb_epochs):

    train_acc_comparison = torch.zeros(nb_epochs)
    test_acc_comparison = torch.zeros(nb_epochs)
    train_loss = torch.zeros(nb_epochs)
    test_loss = torch.zeros(nb_epochs)
    N_train = train.size(0)
    N_test = test.size(0)

    optimizer = SGD(model.param(), eta)

    for epoch in range(nb_epochs):
        train_accuracy = 0
        for batch in range(0, N_train, mini_batch_size):
            output = model.forward(train.narrow(0, batch, mini_batch_size))
            _, predicted_classes = output.max(1)
            loss = optimizer.loss(output, train_target.narrow(0, batch, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += (train_target[batch:batch+mini_batch_size] == predicted_classes).sum().item()

        train_loss[epoch] = loss.item()

        # compute test loss and accuracy without computing the gradients
        output = model.forward(test)
        loss = optimizer.loss(output, test_target)
        test_loss[epoch] = loss.item()
        _, predicted_classes = output.max(1)
        test_accuracy = (test_target == predicted_classes).sum().item()


        # compute accuracy
        train_acc_comparison[epoch] = train_accuracy / N_train
        test_acc_comparison[epoch] = test_accuracy / N_test

    return train_acc_comparison, test_acc_comparison, train_loss, test_loss


def generate_data():
    input = torch.Tensor(size, 2).uniform_(0, 1)
    target = input.sub(torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target
