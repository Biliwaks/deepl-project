from torch import empty

class Module(object):

    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []



'''
Applies a linear transformation to the incoming data: :math:y = xA^T + b
ARGS:
    - weight: the learnable weights
    - bias: the learnable bias
SHAPE:
    - weight: (out_features, in_features)
    - bias: (out_features)

'''
class Linear(Module):

    def __init__(self, in_features, out_features, bias = True):
        self.in_features = in_features
        self.out_features = out_features
        init_bound = math.sqrt(6)/math.sqrt(self.in_features + self.out_features)
        self.weight = torch.distributions.uniform.Uniform(torch.tensor([-init_bound]), torch.tensor([init_bound])).sample(torch.Size([out_features, in_features]))
        if bias:
            self.bias = torch.distributions.uniform.Uniform(torch.tensor([-init_bound]), torch.tensor([init_bound])).sample(torch.Size([out_features]))
        else:
            self.bias = torch.zeros((out_features))

    def forward (self, input):
        return input@self.weight + self.bias

    def backward (self, *gradwrtoutput):
        return

    def param (self):
        return [self.weight, self.bias]

class ReLU(Module):

    def forward (self, *input):
        if input[0] > 0:
            return input[0]
        else:
            return 0
    def backward (self, *gradwrtoutput):
        return
    def param (self):
        return []

class Tanh(Module):

    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []

class Sequential(Module):

    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []
