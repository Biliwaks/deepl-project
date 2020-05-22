import pandas as pd
from termcolor import colored
from project2 import *


def main():

    print(colored('Project 2 running ...', 'cyan'))

    # Requirements given by project2
    input_units = 2
    output_units = 2
    nb_hidden_units = 25
    nb_samples = 1000
    repeats = 25

    results_list = []

    for i in range(repeats):

        train, train_target = generate_data(nb_samples)
        test, test_target = generate_data(nb_samples)

        train_one_hot_target = one_hot_encoding(train_target)
        test_one_hot_target = one_hot_encoding(test_target)

        model = Sequential(Linear(input_units, nb_hidden_units), ReLU(),
                           Linear(nb_hidden_units, nb_hidden_units), ReLU(),
                           Linear(nb_hidden_units, nb_hidden_units), ReLU(),
                           Linear(nb_hidden_units, output_units))

        model, result_list = train_and_evaluate(
            model, train, train_one_hot_target, test, test_one_hot_target, 25, 1e-2, 100)

        results_list.extend(result_list)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results_relu.csv")

    print(colored("Train and Test losses logged at results.csv"))
    print("Train accuracy: {}, Test accuracy: {}".format(
        results_list[-1]["train_accuracy"], results_list[-1]["test_accuracy"]))

    print(colored('Project 2 done', 'cyan'))


if __name__ == "__main__":
    main()
