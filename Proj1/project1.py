import torch
import os
import copy
import dlc_practical_prologue as prologue
from torch import nn
from torch.nn import functional as F
from torch import optim


IMAGE_SIZE = 196
NUM_CLASSES = 10


class ShallowFCNet(nn.Module):
    """
    This class implements a shallow fully connected neural network.
    There are two fully connected layers. The network transforms the input
    to a hidden layer of 120 neurons and reduces the 120 neurons by a second
    layer of size NUM_CLASSES.  Not used in our final report.
    PARAMETERS:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, dropout=0):
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
    """
    This class implements a deep fully connected neural network. Given a number
    of hidden layers and droput, the class constructs a Fully Connected network
    doubling it's size at each layer until the half, and then reducing
    it size by 2 until the end to reach back the input size. A final Fully
    Connected layer is then added to get the desired output size.
    PARAMETERS:
        -nb_layers: number of expanding and contracting layers:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, nb_layers=4, dropout=0):
        super(DeepFCNet, self).__init__()
        self.layers = nn.ModuleList([])
        self.name = f"DeepFCNet({nb_layers})_dropout_{dropout}"
        self.drop = nn.Dropout(dropout)
        acc = IMAGE_SIZE
        if nb_layers % 2 != 0:
            nb_layers = nb_layers - 1
        for l in range(int(nb_layers)):
            if l < nb_layers / 2:
                self.layers.append(nn.Linear(acc, acc * 2))
                acc = acc * 2
            else:
                self.layers.append(nn.Linear(acc, int(acc / 2)))
                acc = int(acc / 2)
        self.layers.append(nn.Linear(IMAGE_SIZE, 10))

    def forward(self, x):
        acc = IMAGE_SIZE
        for l in range(len(self.layers) - 1):
            x = F.relu(self.layers[l](x.view(-1, acc)))
            x = self.drop(x)
            if l < (len(self.layers) - 1) / 2:
                acc = acc * 2
            else:
                acc = int(acc / 2)
        x = self.layers[len(self.layers) - 1](x)

        return x


class BasicCNN(nn.Module):
    """
    This class implements a convolutional neural network. Three succesive
    expanding convolutions are connected by three fully connected layers to
    reach the output size at the end.
    PARAMETERS:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, dropout=0):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN(dropout_{dropout})"

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 1, 14, 14)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 32 * 5 * 5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


class BasicCNN_bn(nn.Module):
    """
    A variant of BasicCNN, which implements also batch normalization after
    each convolutional layer. Not used in our final report.
    PARAMETERS:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, dropout=0):
        super(BasicCNN_bn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=4)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 32, kernel_size=4)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(dropout)
        self.name = f"BasicCNN_bn_dropout_{dropout}"

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x.view(-1, 1, 14, 14))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.fc1(x.view(-1, 64 * 5 * 5)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


class LeNet4(nn.Module):
    """
    Class implementing the LeNet4 architecture. To adapt to our input size a
    padding of 9 is added to the images since the oiginal LeNet4 is used for
    28x28 images.
    PARAMETERS:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, dropout=0):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=9)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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
    """
    Class implementing the LeNet5 architecture. It is a variant of the of the
    LeNet4 model, with an additional Fully Connected layer. To adapt to our
    input size a padding of 9 is added to the images since the original LeNet5
    is used for 28x28 images.
    PARAMETERS:
        -dropout: the value of the dropout to give to the model
    """

    def __init__(self, dropout=0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=9)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.drop = nn.Dropout(dropout)
        self.name = f"LeNet5_dropout_{dropout}"

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
    """
    Class implementing a residual block combined in Residual Nets. It is a
    small block composed of two convolutional layers. The input is also added
    to the output of the final batch normalization after the second convolution.
    The output is the same size as the input. Not used in our final report.
    PARAMETERS:
        -nb_channels: Number of channels for the convolutions
        -kernel_size: Kernel size for the convolutions
    """

    def __init__(self, nb_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    """
    Class implementing a residual net combining in Residual blocks.
    Not used in our final report.
    PARAMETERS:
        -nb_channels: Number of channels for the convolutions
        -kernel_size: Kernel size for the convolutions
        -nb_blocks: number of residual blocks to add to the network.
    """

    def __init__(self, nb_channels, kernel_size, nb_blocks):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(1, nb_channels, kernel_size=1)
        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size=14)
        self.fc = nn.Linear(nb_channels, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x.view(-1, 1, 14, 14)))
        x = self.resblocks(x)
        x = F.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseNet_WS(nn.Module):
    """
    Implements a tandem network with weight sharing called siamese networks.
    Computes two outputs using the same model and combines them to obtain a
    result from the two inputs using two fully connected layers. Uses the same
    instance of the subnetwork to perform weight sharing.
    PARAMETERS:
        -base_model: Model to use to do the auxiliary tasks in parallel
        -dropout: dropout value to use at the combination of outputs
    """

    def __init__(self,  base_model, dropout=0):
        super(SiameseNet_WS, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.base_model = base_model
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)
        self.name = f"SiameseNet_WS_dropout_{dropout}"

    def forward(self, img1, img2):
        # foward pass of input 1
        output1 = self.base_model(img1)
        # forward pass of input 2
        output2 = self.base_model(img2)

        result = torch.cat((output1, output2), dim=1, out=None)
        result = F.relu(self.fc1(result))
        result = self.drop(result)
        result = self.fc2(result)

        return output1, output2, result


class SiameseNet_noWS(nn.Module):
    """
    Implements a tandem network without weight sharing .Computes two outputs
    using two different instances of the same model and combines them to
    obtain a result from the two inputs using two fully connected layers.
    PARAMETERS:
        -base_model1: first model instance that trains the first input
        -base_model2: second model instance that trains the second input
        -dropout: dropout value to use at the combination of outputs
    """

    def __init__(self, base_model1, base_model2, dropout=0):
        super(SiameseNet_noWS, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)
        self.name = f"SiameseNet_noWS_dropout_{dropout}"

    def forward(self, img1, img2):
        # foward pass of input 1
        output1 = self.base_model1(img1)
        # foward pass of input 2
        output2 = self.base_model2(img2)

        result = torch.cat((output1, output2), dim=1, out=None)
        result = F.relu(self.fc1(result))
        result = self.drop(result)
        result = self.fc2(result)

        return output1, output2, result


"""
Dictionary having key equal to the optimizer_name and return the optimizer from
the torch library
PARAMETERS:
    - parameters: the parameters of the model
    - eta: the learning rate used for training
    - momentum: the momentum used in the train (if the optimizer does not have a
    momentum parameter then the momentum is ignored)
RETURN:
    - the optim library of pytorch representing the optimizer from the key
"""
optimizer_methods = {
    'SGD': (lambda parameters, eta, momentum: optim.SGD(parameters(), eta, momentum=momentum)),
    'Adam': (lambda parameters, eta, momentum: optim.Adam(parameters(), eta))
}


def train_model(model, train, train_target, train_classes, test, test_target, test_classes,
                mini_batch_size, eta, criterion, nb_epochs, momentum, optimizer_name,
                weight_sharing, auxiliary, dropout, result_list):
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

    N_train = train[0].size(0)

    if weight_sharing:
        siamese_model = SiameseNet_WS(model, dropout)
    else:
        # Get a new instance of the model
        model2 = copy.deepcopy(model)
        siamese_model = SiameseNet_noWS(model, model2, dropout)

    optimizer = optimizer_methods[optimizer_name](
        siamese_model.parameters, eta, momentum)

    for epoch in range(nb_epochs):
        print(epoch)
        train_accuracy = 0
        running_loss = 0
        for batch in range(0, N_train, mini_batch_size):
            out1, out2, results = siamese_model(train[0].narrow(
                0, batch, mini_batch_size), train[1].narrow(0, batch, mini_batch_size))

            # Compute the primary loss
            loss_results = criterion(
                results, train_target.narrow(0, batch, mini_batch_size))

            # Compute the auxiliary losses
            loss1 = criterion(out1, train_classes[0].narrow(
                0, batch, mini_batch_size))
            loss2 = criterion(out2, train_classes[1].narrow(
                0, batch, mini_batch_size))

            # Compute the final loss which is a weighted sum of the lossess
            loss = auxiliary[0] * (loss1 + loss2) + auxiliary[1] * loss_results

            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out1, out2, results = siamese_model(test[0], test[1])
            loss_results = criterion(results, test_target)

            loss1 = criterion(out1, test_classes[0])
            loss2 = criterion(out2, test_classes[1])
            loss = auxiliary[0] * (loss1 + loss2) + auxiliary[1] * loss_results

            # Compute train and test accuracy
            test_accuracy = compute_accuracy(
                siamese_model, test[0], test[1], test_target)
            train_accuracy = compute_accuracy(
                siamese_model, train[0], train[1], train_target)

        # This line is for our plots
        result_list.append({"epoch": epoch, "model": model.__class__.__name__, "weight sharing": weight_sharing,
                            "auxiliary": auxiliary,
                            "train accuracy": train_accuracy,
                            "train loss": running_loss,
                            "test accuracy": test_accuracy,
                            "test loss": test_loss.item()})

    return siamese_model


def compute_accuracy(siamese_model, input1, input2, target):
    """
        Computes the prediction of the model and compute the accuracy of the
        model with respect to the primary loss.
        PARAMETERS:
            -siamese_model: the tandem model to compute the accuracy.
            -input1: input of the first image
            -input2: input of the second image
            -target: the comparison target to achieve
        RETURNS:
            The accuracy of the model as a percentage between 0 and 1
    """
    N = input1.size(0)
    out1, out2, results = siamese_model(input1, input2)
    _, predictions = results.max(1)
    correct_predictions = (predictions == target).sum().item()

    return correct_predictions / N


def generate_data():
    """
    Generates the pair of images using the helper methods provided in the
    prologue. Outputs the train and test images as pairs of size (Nx14x14)
    and outputs the train and test classes as pairs of size (Nx1). The train
    and test target are tensors of size (N x 1).
    """

    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        1000)

    train = (train_input[:, 0, :, :], train_input[:, 1, :, :])
    test = (test_input[:, 0, :, :], test_input[:, 1, :, :])

    train_classes = (train_classes[:, 0], train_classes[:, 1])
    test_classes = (test_classes[:, 0], test_classes[:, 1])

    return train, test, train_target, test_target, train_classes, test_classes
