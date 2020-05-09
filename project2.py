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
        self.previous_layer = None
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

    def forward (self, input):
        zeroes = torch.full(input.size(), 0.0, dtype = float )
        return torch.where(input > 0, input.float(), zeroes.float())
    def backward (self, gradwrtoutput):
        ones = torch.ones(gradwrtoutput.size())
        zeroes = torch.full(gradwrtoutput.size(), 0.0, dtype = float )
        return torch.where(gradwrtoutput > 0, ones.float(), zeroes.float())
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
        x = input
        for module in self.modules:
            x = module.forward(x)
        return input

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
        return (prediction - target[:, None]).pow(2).sum()

    def grad(self, prediction, target):
        return 2*(prediction - target[:, None])

def normalize_data(x):
    mean, std =  x.mean(), x.std()
    x.sub_(mean).div_(std)


def train_model(model, train, train_target, test, test_target,
                mini_batch_size, eta, nb_epochs, normalize = True):

    train_accuracy = torch.zeros(nb_epochs)
    test_accuracy = torch.zeros(nb_epochs)
    train_loss = torch.zeros(nb_epochs)
    test_loss = torch.zeros(nb_epochs)
    N_train = train.size(0)
    N_test = test.size(0)

    if normalize:
        normalize_data(train)
        normalize_data(test)


    optimizer = SGD(model.param(), eta)
    MSE = LossMSE()

    for epoch in range(nb_epochs):

        nb_correct_classes_tr = 0
        loss = 0

        for batch in range(0, N_train, mini_batch_size):
            optimizer.zero_grad()
            output = model.forward(train.narrow(0, batch, mini_batch_size))
            loss += MSE.loss(output, train_target.narrow(0, batch, mini_batch_size))
            loss_grad = MSE.grad(output, train_target.narrow(0, batch, mini_batch_size))
            _, predicted_classes = output.max(1)

            model.backward(loss_grad)

            optimizer.step()

            nb_correct_classes_tr += (train_target[batch:batch+mini_batch_size] == predicted_classes).sum().item()

        train_loss[epoch] = loss

        # compute test loss and accuracy without computing the gradients
        output = model.forward(test)
        loss = MSE.loss(output, test_target)
        test_loss[epoch] = loss.item()
        _, predicted_classes = output.max(1)
        nb_correct_classes_te = (test_target == predicted_classes).sum().item()


        # compute accuracy
        train_accuracy[epoch] = nb_correct_classes_tr / N_train
        test_accuracy[epoch] = nb_correct_classes_te / N_test

    return train_accuracy, test_accuracy, train_loss, test_loss


def generate_data(size):
    input = torch.Tensor(size, 2).uniform_(0, 1)
    target = input.sub(torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target

def convert_to_one_hot(target):
    tmp = torch.FloatTensor(target.size(0), 2).fill_(0)
    for k in range(0, target.size(0)):
        tmp[k, target[k]] = 1
    return tmp



train, train_target = generate_data(800)
test, test_target = generate_data(200)


# Requirements given by project2
input_units = 2
output_units = 2
nb_hidden_units = 25

project2Net = Sequential(Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25, 25), ReLU(), Linear(25,2))
train_model(project2Net, train, train_target, test, test_target, 100, 1e-1, 25)






# from torch import empty
# import torch
# import math
#
# class Module(object):
#
#     def forward (self, *input):
#         raise NotImplementedError
#     def backward (self, *gradwrtoutput):
#         raise NotImplementedError
#     def param (self):
#         return []
#
#
#
# '''
# Applies a linear transformation to the incoming data: y = xA^T + b
# ARGS:
#     - weight: the learnable weights
#     - bias: the learnable bias
# SHAPE:
#     - weight: (out_features, in_features)
#     - bias: (out_features)
#
# '''
# class Linear(Module):
#
#     def __init__(self, in_features, out_features, bias = True):
#         init_bound = 1/math.sqrt(in_features)
#         self.weight = torch.Tensor(out_features, in_features).uniform_(-init_bound, init_bound)
#         if bias:
#             self.bias = torch.Tensor(out_features).uniform_(-init_bound, init_bound)
#         else:
#             self.bias = torch.zeros(out_features)
#         self.grad_weight = torch.zeros(self.weight.size())
#         self.grad_bias = torch.zeros(self.bias.size())
#
#     def forward (self, input):
#         self.input = input
#         return self.weight.mv(self.input) + self.bias
#
#     def backward (self, gradwrtoutput):
#         self.grad_weight.add_(gradwrtoutput.mm(self.input.t()))
#         self.grad_bias.add_(gradwrtoutput.sum(0))
#         grad_input = gradwrtoutput.mv(self.weight)
#         return grad_input
#
#     def param (self):
#         return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]
#
# class ReLU(Module):
#
#     def forward (self, input):
#         zeroes = torch.full(input.size, 0.0, dtype = float )
#         return torch.where(input > 0, input, zeroes)
#     def backward (self, gradwrtoutput):
#         ones = torch.ones(input.size())
#         zeroes = torch.full(input.size(), 0.0, dtype = float )
#         return torch.where(gradwrtoutput > 0, ones, zeroes)
#     def param (self):
#         return [(None, None)]
#
# class Tanh(Module):
#
#     def forward (self, input):
#         return input.tanh()
#     def backward (self, gradwrtoutput):
#         return 1 - (gradwrtoutput.tanh()).pow(2)
#     def param (self):
#         return [(None, None)]
#
# class Sequential(Module):
#
#     def __init__(self, *modules):
#         self.modules = list(modules)
#
#     def forward (self, input):
#         for module in self.modules:
#             input = module.forward(input)
#         return input
#
#     def backward (self, gradwrtoutput):
#         for module in reversed(self.modules):
#             gradwtroutput = module.backward(gradwrtoutput)
#         return gradwtroutput
#
#     def param (self):
#         parameters = []
#         for module in self.modules:
#             parameters.append(module.param())
#         return parameters
#
# class SGD():
#
#     def __init__(self, parameters, eta):
#         self.parameters = parameters
#         if eta < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(eta))
#         else:
#             self.eta = eta
#
#     def step(self):
#         for module in self.parameters:
#             for param, grad_param in module:
#                 if (param is not None and grad_param is not None):
#                     param.sub_(self.eta * grad_param)
#
#
#     def zero_grad(self):
#         for module in self.parameters:
#             for param, grad_param in module:
#                 if (param is not None and grad_param is not None):
#                     grad_param.zero_()
#
#
# class LossMSE():
#     def loss(self, prediction, target):
#         return (target-prediction).pow(2).sum()
#
#     def grad(self, prediction, target):
#         return 2*(prediction - target)
#
#
#
#
# def train_model(model, train, train_target, test, test_target,
#                 mini_batch_size, eta, nb_epochs):
#
#     train_acc_comparison = torch.zeros(nb_epochs)
#     test_acc_comparison = torch.zeros(nb_epochs)
#     train_loss = torch.zeros(nb_epochs)
#     test_loss = torch.zeros(nb_epochs)
#     N_train = train.size(0)
#     N_test = test.size(0)
#
#     optimizer = SGD(model.param(), eta)
#
#     for epoch in range(nb_epochs):
#         train_accuracy = 0
#         for batch in range(0, N_train, mini_batch_size):
#             output = model.forward(train.narrow(0, batch, mini_batch_size))
#             _, predicted_classes = output.max(1)
#             loss = optimizer.loss(output, train_target.narrow(0, batch, mini_batch_size))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_accuracy += (train_target[batch:batch+mini_batch_size] == predicted_classes).sum().item()
#
#         train_loss[epoch] = loss.item()
#
#         # compute test loss and accuracy without computing the gradients
#         output = model.forward(test)
#         loss = optimizer.loss(output, test_target)
#         test_loss[epoch] = loss.item()
#         _, predicted_classes = output.max(1)
#         test_accuracy = (test_target == predicted_classes).sum().item()
#
#
#         # compute accuracy
#         train_acc_comparison[epoch] = train_accuracy / N_train
#         test_acc_comparison[epoch] = test_accuracy / N_test
#
#     return train_acc_comparison, test_acc_comparison, train_loss, test_loss
#
#
# def generate_data():
#     input = torch.Tensor(size, 2).uniform_(0, 1)
#     target = input.sub(torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
#     return input, target
