import dlc_practical_prologue as prologue
import json
from torch import  nn
import torch

from project1 import *

def main():
    print('Project 1')
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

    train = torch.cat(( train_input[:, 0, :, :], train_input[:, 1, :, :]), 0)
    test = torch.cat((test_input[:, 0, :, :], test_input[:, 1, :, :]), 0)

    train_classes = torch.cat((train_classes[:, 0], train_classes[:, 1]), 0)
    test_classes = torch.cat((test_classes[:, 0], test_classes[:, 1]), 0)

    models = [ShallowFCNet(), DeepFCNet(), BasicCNN(), BasicCNN_bn(), LeNet4(), LeNet5(), ResNet()]
    optimizers = ['SGD']
    dropouts = [0, 0.25]
    criterions = [nn.CrossEntropyLoss(), nn.MultiMarginLoss()]
    epochs = [1]

    all_results = []

    for model in models:
        for optimizer in optimizers:
            for criterion in criterions:
                for dropout in dropouts:
                    for epoch in epochs:
                        all_results = train_test(model, train, test, train_classes,
                                    test_classes, train_target, test_target, 100,
                                    criterion, epoch, optimizer_name = optimizer)

    with open('comparison_models.json', 'w') as json_file:
        json.dump(all_results[0], json_file)
    print(all_results)
    print('Project 1 done')

def generate_input(size):
    input = torch.Tensor(size, 2).uniform_(0, 1)
    target = input.sub(torch.tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target


if __name__ == "__main__":
    main()
