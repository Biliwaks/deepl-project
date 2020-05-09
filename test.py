import dlc_practical_prologue as prologue
import json
from torch import  nn
import torch
import os

# il va falloir les importer séparement 
from project1 import *
from project2 import *

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
    PATH =  os.path.dirname(os.path.abspath(__file__))

    for model in models:
        for optimizer in optimizers:
            for criterion in criterions:
                for dropout in dropouts:
                    for epoch in epochs:
                        all_results = train_test(model, train, test, train_classes,
                                    test_classes, train_target, test_target, 100,
                                    criterion, epoch, optimizer_name = optimizer)

                        torch.save(model.state_dict(), PATH)

    with open('comparison_models.json', 'w') as json_file:
        json.dump(all_results[0], json_file)
    print(all_results)

    print('Project 1 done')
    print('')


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

    train_model(model, train, train_one_hot_target, test, test_one_hot_target, 25, 1e-4, 100)






if __name__ == "__main__":
    main()
