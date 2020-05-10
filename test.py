import dlc_practical_prologue as prologue
import json
import torch
import os
from torch import  nn
from termcolor import colored

# il va falloir les importer séparement
from project1 import *
from project2 import *

def main():
    print(colored('Project 1 running ...', 'cyan'))
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

    train = torch.cat(( train_input[:, 0, :, :], train_input[:, 1, :, :]), 0)
    test = torch.cat((test_input[:, 0, :, :], test_input[:, 1, :, :]), 0)

    train_classes = torch.cat((train_classes[:, 0], train_classes[:, 1]), 0)
    test_classes = torch.cat((test_classes[:, 0], test_classes[:, 1]), 0)

    models = [ShallowFCNet(), DeepFCNet(), BasicCNN(), BasicCNN(dropout= 0.25), BasicCNN_bn(), BasicCNN_bn(dropout=0.25), LeNet4(), LeNet4(dropout=0.25), LeNet5(), LeNet5(dropout=0.25), ResNet(), ResNet(dropout = 0.25)]
    optimizers = ['SGD', 'Adam']
    criterions = [nn.CrossEntropyLoss(), nn.MultiMarginLoss()]
    epochs = [25, 50, 100]

    all_results = []

    for model in models:
        for optimizer in optimizers:
            for criterion in criterions:
                for epoch in epochs:
                    all_results.append(train_test(model, train, test, train_classes,
                                test_classes, train_target, test_target, 100,
                                criterion, epoch, optimizer_name = optimizer))
                    model_name =  "{}_{}".format(model.name, criterion.__class__.__name__ )
                    save_model_all(model, model_name, epoch)
                    print("Model saved.")

    with open('comparison_models.json', 'w') as json_file:
        json.dump(all_results, json_file)

    print(colored('Project 1 done', 'cyan'))

    print(colored('Project 2 running', 'cyan'))


    train, train_target = generate_data(1000)
    test, test_target = generate_data(1000)

    train_one_hot_target = one_hot_encoding(train_target)
    test_one_hot_target = one_hot_encoding(test_target)


    # Requirements given by project2
    input_units = 2
    output_units = 2
    nb_hidden_units = 25

    model = Sequential(Linear(input_units, nb_hidden_units), Tanh(),
                             Linear(nb_hidden_units, nb_hidden_units), Tanh(),
                             Linear(nb_hidden_units, nb_hidden_units), Tanh(),
                             Linear(nb_hidden_units, output_units))


    project2_results = train_and_evaluate(model, train, train_one_hot_target, test, test_one_hot_target, 25, 1e-4, 100)

    with open('project2_results.json', 'w') as json_file:
        json.dump(project2_results, json_file)

    print(colored('Project 2 done', 'cyan'))



if __name__ == "__main__":
    main()
