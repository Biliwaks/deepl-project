import dlc_practical_prologue as prologue
import torch
import os
from torch import  nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor
from torch import nn


IMAGE_SIZE = 196
NUM_CLASSES = 10

class ShallowFCNet(nn.Module):
    def __init__(self, dropout = 0):
        super(ShallowFCNet, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, 120)
        self.fc2 = nn.Linear(120, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"ShallowFCNet_dropout_{dropout}"

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, IMAGE_SIZE)))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class DeepFCNet(nn.Module):
    def __init__(self, nb_layers=4, dropout = 0):
        super(DeepFCNet, self).__init__()
        self.layers = nn.ModuleList([])
        self.name = f"DeepFCNet({nb_layers})"
        self.drop = nn.Dropout(dropout)
        acc = IMAGE_SIZE
        if nb_layers % 2 !=0:
            nb_layers = nb_layers - 1
        for l in range(nb_layers):
            if l < nb_layers/2:
                self.layers.append(nn.Linear(acc, acc*2))
                acc = acc*2
            else:
                self.layers.append(nn.Linear(acc, int(acc/2)))
                acc = int(acc/2)
        self.layers.append(nn.Linear(IMAGE_SIZE, 10))

    def forward(self, x):
        acc = IMAGE_SIZE
        for l in range(len(self.layers)-1):
            x = F.relu(self.layers[l](x.view(-1, acc)))
            x = self.drop(x)
            if l < (len(self.layers)-1)/2:
                acc = acc*2
            else:
                acc = int(acc/2)
        x = self.layers[len(self.layers)-1](x)

        return x

class BasicCNN(nn.Module):
    def __init__(self, dropout = 0):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN(dropout_{dropout})"

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 14, 14)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 32*5*5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

class BasicCNN_bn(nn.Module):
    def __init__(self, dropout = 0):
        super(BasicCNN_bn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 32, kernel_size=4)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.comp1 = nn.Linear(20, 150)
        self.comp2 = nn.Linear(150, 2)
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN_bn_dropout_{dropout}"

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x.view(-1, 1, 14, 14))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.fc1(x.view(-1, 64*5*5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x



class LeNet4(nn.Module):
    def __init__(self, dropout = 0):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding = 9)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"LeNet4_dropout_{dropout}"

    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 14, 14))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self, dropout = 0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding = 9)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"LeNet45, dropout = {dropout}"

    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 14, 14))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, dropout = 0):
        super(ResBlock, self).__init__()
        self.conv2 = nn.Conv2d(6, 32, kernel_size=4)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(dropout)
        self.conv2_bis = nn.Conv2d(6, 64, kernel_size=1)
        self.avg = nn.AvgPool2d(kernel_size = 2)
        self.max = nn.MaxPool2d(kernel_size = 2)
        self.conv2_bis_bn = nn.BatchNorm2d(64)
    def forward(self, x):
        y = self.conv2_bn(self.conv2(x))
        y = F.relu(y)
        y = self.conv3_bn(self.conv3(y))
        y += F.relu(self.conv2_bis_bn(self.avg(self.conv2_bis(x)))) + F.relu(self.conv2_bis_bn(self.max(self.conv2_bis(x))))
        y = F.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self, dropout = 0):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.resblock = ResBlock(dropout)
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"ResNet_dropout_{dropout}"
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x.view(-1, 1, 14, 14))))
        x = self.resblock(x)
        x = F.relu(self.fc1(x.view(-1, 64*5*5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


class SiameseNet_WS(nn.Module):
    def __init__(self,  base_model, dropout = 0):
        super(SiameseNet_WS, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.base_model = base_model
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)
        self.name = f"SiameseNet_WS_with_BasicCNN_bn_dropout_{dropout}"

    def forward(self, img1, img2):
        # foward pass of input 1
        output1 = self.base_model(img1)
        # foward pass of input 2
        output2 = self.base_model(img2)

        result = torch.cat((output1,output2), dim=1, out=None)
        result = F.relu(self.fc1(result))
        result = self.drop(result)
        result = self.fc2(result)

        return output1, output2, result


class SiameseNet_noWS(nn.Module):
    def __init__(self, base_model1, base_model2, dropout = 0):
        super(SiameseNet_noWS, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)
        self.name = f"SiameseNet_noWS_with_BasicCNN_bn_dropout_{dropout}"

    def forward(self, img1, img2):
        # foward pass of input 1
        output1 = self.base_model1(img1)
        # foward pass of input 2
        output2 = self.base_model2(img2)

        result = torch.cat((output1,output2), dim=1, out=None)
        result = F.relu(self.fc1(result))
        result = self.drop(result)
        result = self.fc2(result)

        return output1, output2, result

class ModeleJo(nn.Module):
    def __init__(self, dropout = 0):
        super(ModeleJo, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop = nn.Dropout(dropout)
        self.name = f"LeNet45, dropout = {dropout}"

    def forward(self, x):
        x = self.conv1(x.view(-1, 1, 14, 14))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.drop(x)
        x = self.fc2(x)
        return x



"""
Dictionnary having key equal to the optimizer_name and return the optimizer from the torch library
PARAMETERS:
    - parameters: the parameters of the model
    - eta: the learning rate used for training
    - momentum: the momentum used in the train (if the optimizer does not have a
    momentum parameter then the momentum is ignored)
RETURN:
    - the optim library of pytorch representing the optimizer from the key
"""
optimizer_methods = {
    'SGD': (lambda parameters, eta, momentum: optim.SGD(parameters(), eta, momentum = momentum)),
    'Adam': (lambda parameters, eta, momentum: optim.Adam(parameters(), eta))
}


"""
This method trains the model, along the run it stores the train loss and train digit accuracy for every epoch.
It also stores the test loss and the test digit accuracy for every epoch into torch tensor.
The digit accuracy does not represent the goal of the project but rather if the first digit
is equal to the second digit.
PARAMETERS:
    - model: nn model
    - train: training data
    - train_classes: classes from the training data
    - test: test data
    - test_classes: classes from the test data
    - mini_batch_size: the size of each batch
    - eta: the learning rate used in the training
    - criterion: nn loss
    - nb_epochs: the number of epochs used in the training process
    - momentum: momentum used for the optimizer
    - optimizer_name: the name of the optimizer used (string)
RETURN:
    - train_accuracy: torch tensor representing the train digit accuracy at every epoch
    - test_accuracy: torch tensor representing the test digit accuracy at every epoch
    - train_loss: torch tensor representing the train loss at every epoch
    - test_loss: torch tensor representing the test loss at every epoch
"""
def train_model(model, train, train_target, train_classes, test, test_target, test_classes,
                mini_batch_size, eta, criterion, nb_epochs, momentum, optimizer_name,
                weight_sharing, auxiliary_loss_param, dropout):

    N_train = train[0].size(0)
    N_test = test[0].size(0)

    if weight_sharing:
        siamese_model = SiameseNet_WS(model, dropout)
    else:
        # Get a new instance of the model
        model2 = type(model)()
        siamese_model = SiameseNet_noWS(model, model2, dropout)

    optimizer = optimizer_methods[optimizer_name](siamese_model.parameters, eta, momentum)

    for epoch in range(nb_epochs):
        train_accuracy = 0
        for batch in range(0, N_train, mini_batch_size):

            out1, out2, results = siamese_model(train[0].narrow(0, batch, mini_batch_size), train[1].narrow(0, batch, mini_batch_size))


            # Compute loss according to the auxiliary parameters
            loss1 = criterion(out1, train_classes[0].narrow(0, batch, mini_batch_size))
            loss2 = criterion(out2, train_classes[1].narrow(0, batch, mini_batch_size))
            loss = nn.BCELoss()
            loss_results = criterion(results, train_target.narrow(0, batch, mini_batch_size))

            final_loss = auxiliary_loss_param[0]*(loss1 + loss2) + auxiliary_loss_param[1]*loss_results

            train_accuracy += compute_accuracy(siamese_model, train[0], train[1], train_target)

            # Backward and optimize
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

    return siamese_model

def one_hot_encoding(target):
    onehot = torch.zeros(target.size(0), 2).fill_(0)
    onehot[range(onehot.shape[0]), target]=1
    return onehot


def compute_accuracy(siamese_model, input1, input2, target):
    N = input1.size(0)
    out1, out2, results = siamese_model(input1, input2)
    _, predictions = results.max(1)
    correct_predictions = (predictions == target).sum().item()
    return correct_predictions / N


def train_test(model, mini_batch_size, criterion,
             nb_epochs, eta = 1e-1, momentum = 0, optimizer_name = 'SGD',
             repeats = 1, weight_sharing = True, auxiliary_loss = (1, 1), dropout = 0):

    train_accuracy = torch.zeros(repeats, 1)
    test_accuracy = torch.zeros(repeats, 1)
    for i in range(repeats):
        train, test, train_target, test_target, train_classes, test_classes = generate_data()

        siamese_model = train_model(model, train, train_target, train_classes,
            test, test_target, test_classes, mini_batch_size, eta, criterion, nb_epochs, momentum,
            optimizer_name, weight_sharing, auxiliary_loss, dropout)


        train_accuracy[i] = compute_accuracy(siamese_model, train[0], train[1], train_target)
        test_accuracy[i] = compute_accuracy(siamese_model, test[0], test[1], test_target)

    return test_accuracy.mean(), test_accuracy.std()



def generate_data():
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

    train = (train_input[:, 0, :, :], train_input[:, 1, :, :])
    test = (test_input[:, 0, :, :], test_input[:, 1, :, :])

    train_classes = (train_classes[:, 0], train_classes[:, 1])
    test_classes = (test_classes[:, 0], test_classes[:, 1])

    return train, test, train_target, test_target, train_classes, test_classes



def save_model_all(model, model_name, epoch):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir("models/"):
        os.makedirs("models/")
    save_prefix = os.path.join("models/", model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()

"""
Enables to load the saved model
PARAMETERS:
    - model: the model previously saved
    - path: the path where the model was saved
"""
def load_saved_model(model, path):
    model_out = model
    model_out.load_state_dict(torch.load(path))
    return model_out.eval()
