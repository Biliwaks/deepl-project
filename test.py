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

    epochs = 25
    models = [ShallowFCNet(), BasicCNN_bn(0.25)]
    weight_sharings = [False, True]
    auxiliary_losses = [(0, 1), (1, 1)]
    dropout = 0.25

    results = []

    for model in models:
        for weight_sharing in weight_sharings:
            for auxiliary_loss in auxiliary_losses:
                x = train_test(BasicCNN_bn(dropout), 100,
                            nn.CrossEntropyLoss(), epochs, dropout = dropout, optimizer_name = 'SGD',
                            weight_sharing = weight_sharing, auxiliary_loss = auxiliary_loss)
                print(x)
                results.append(x)

    print(results)

    print(colored('Project 1 done', 'cyan'))

    print(colored('Project 2 running ...', 'cyan'))

    # Requirements given by project2
    input_units = 2
    output_units = 2
    nb_hidden_units = 25
    nb_samples = 1000

    train, train_target = generate_data(nb_samples)
    test, test_target = generate_data(nb_samples)

    train_one_hot_target = one_hot_encoding(train_target)
    test_one_hot_target = one_hot_encoding(test_target)


    model = Sequential(Linear(input_units, nb_hidden_units), Tanh(),
                       Linear(nb_hidden_units, nb_hidden_units), Tanh(),
                       Linear(nb_hidden_units, nb_hidden_units), Tanh(),
                       Linear(nb_hidden_units, output_units))


    project2_results = train_and_evaluate(model, train, train_one_hot_target, test, test_one_hot_target, 25, 1e-2, 100)

    with open('project2_results.json', 'w') as json_file:
        json.dump(project2_results, json_file)

    print(colored('Project 2 done', 'cyan'))


if __name__ == "__main__":
    main()
