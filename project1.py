import torch
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
        self.name = f"ShallowFCNet, dropout = {dropout}"

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
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN(dropout = {dropout})"

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
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN with batch normalization, dropout = {dropout}"

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
        self.fc2 = nn.Linear(120, 10)
        self.drop = nn.Dropout(dropout)
        self.name = f"LeNet4, dropout = {dropout}"

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
        self.fc3 = nn.Linear(84, 10)
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
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(dropout)
        self.name = f"Residual network inspired from BasicCNN_bn, dropout = {dropout}"
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x.view(-1, 1, 14, 14))))
        x = self.resblock(x)
        x = F.relu(self.fc1(x.view(-1, 64*5*5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


optimizer_methods = {
    'SGD': (lambda parameters, eta, momentum: optim.SGD(parameters(), eta, momentum = momentum)),
    'Adam': (lambda parameters, eta, momentum: optim.Adam(parameters(), eta))
}

def train_model(model, train, train_classes, test, test_classes,
                mini_batch_size, eta, criterion, nb_epochs, momentum, optimizer_name):

    train_accuracy = torch.zeros(nb_epochs)
    test_accuracy = torch.zeros(nb_epochs)
    train_loss = torch.zeros(nb_epochs)
    test_loss = torch.zeros(nb_epochs)
    N_train = train.size(0)
    N_test = test.size(0)

    optimizer = optimizer_methods[optimizer_name](model.parameters, eta, momentum)

    for epoch in range(nb_epochs):
        correct_train_digits = 0
        for batch in range(0, N_train, mini_batch_size):
            output = model(train.narrow(0, batch, mini_batch_size))
            _, predicted_classes = output.max(1)
            loss = criterion(output, train_classes.narrow(0, batch, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_train_digits += (train_classes[batch:batch+mini_batch_size] == predicted_classes).sum().item()

        train_loss[epoch] = loss.item()
        with torch.no_grad():
            output = model(test)
            loss = criterion(output, test_classes)
            test_loss[epoch] = loss.item()
            _, predicted_classes = output.max(1)
            correct_test_digits = (test_classes == predicted_classes).sum().item()


        # compute accuracy
        train_accuracy[epoch] = correct_train_digits / N_train
        test_accuracy[epoch] = correct_test_digits / N_test

    return train_accuracy, test_accuracy, train_loss, test_loss


def compute_project_accuracy(model, input1, input2, target):
    output1 = model(input1)
    output2 = model(input2)
    _, predicted_classes1 = output1.max(1)
    _, predicted_classes2 = output2.max(1)

    nb_correct_project = (target == (predicted_classes1 <= predicted_classes2)).sum().item()


    return float(nb_correct_project / target.size(0))



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

def train_test(model, train, test, train_classes, test_classes,
            train_target, test_target, mini_batch_size, criterion,
             nb_epochs, eta = 1e-2, momentum = 0.9, optimizer_name = 'SGD', repeats = 25):
    all_results = []

    N =  int(len(train)/2)
    train_comparison = torch.zeros(repeats,1)
    test_comparison  = torch.zeros(repeats,1)

    train_loss = torch.zeros(repeats, nb_epochs)
    test_loss = torch.zeros(repeats, nb_epochs)

    train_acc = torch.zeros(repeats, nb_epochs)
    test_acc = torch.zeros(repeats, nb_epochs)


    for i in range(repeats):
        model.apply(weights_init)

        train_acc[i], test_acc[i], train_loss[i], test_loss[i] = train_model(model, train, train_classes,
            test, test_classes, mini_batch_size, eta, criterion, nb_epochs, momentum,
            optimizer_name)

        # plot_accuracy(train_comparison[i], test_comparison[i], nb_epochs)

        train_comparison[i] = compute_project_accuracy(model, train[: N], train[N: ], train_target)
        test_comparison[i] = compute_project_accuracy(model, test[: N], test[N: ], test_target)

    all_results.append({"Model": model.name, "Optimizer": optimizer_name , "Epochs": nb_epochs, "Eta": eta, "Train Accuracy Mean": train_comparison.mean().item(),"Test Accuracy Mean": test_comparison.mean().item(), "Train Accuracy Std":  train_acc.std().item(), "Test Accuracy Std": test_acc.std().item(), "Digit acc table":     train_acc.mean(axis= 0)
            , "test Digit Accuracy Table":     test_acc.mean(axis= 0)})

    return all_results
