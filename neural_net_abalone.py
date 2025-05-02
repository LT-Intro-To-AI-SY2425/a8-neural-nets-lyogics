from typing import Tuple
from neural import NeuralNet
from neural_net_UCI_data import parse_line


with open("abalone_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

abann = NeuralNet(8, 0, 1)
abann.train(training_data, learning_rate=0.7, iters=10000, print_interval=1000)

# F = 0.1, M = 0.3, I = 0.2
# features from left to right: sex, length, diameter, whole height, whole weight, weight of meat, gut weight after bleeding, shell weight after drying 
test_data = [
    ([0.2, 0.123, 0.941, 0.322, 0.632, 0.101, 0.021, 0.91], [20]),
    ([0.3, 0.32, 0.18, 0.291, 0.234, 0.41, 0.01, 0.72], [8]),
    ([0.1, 0.302, 0.148, 0.698, 0.251, 0.28, 0.61, 0.82], [15]),
    ([0.1, 0.42, 0.186, 0.421, 0.765, 0.76, 0.53, 0.28], [6]),
    ([0.3, 0.27, 0.671, 0.931, 0.892, 0.522, 0.19, 0.219], [17]),
    ([0.2, 0.189, 0.722, 0.367, 0.124, 0.401, 0.13, 0.92], [12]),
    ([0.3, 0.82, 0.09, 0.458, 0.41, 0.51, 0.954, 0.48], [9]),
    ([0.3, 0.96, 0.102, 0.261, 0.62, 0.472, 0.321, 0.12], [10]),
    ([0.2, 0.34, 0.22, 0.212, 0.89, 0.195, 0.745, 0.732], [7]),
    ([0.1, 0.75, 0.31, 0.875, 0.13, 0.839, 0.987, 0.42], [13])
    ]
# intended output is number of rings

for i in abann.test_with_expected(test_data):
    print(f"desired: {i[1]}, actual: {i[2]}")   
