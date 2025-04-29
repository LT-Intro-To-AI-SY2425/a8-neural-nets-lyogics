from typing import Tuple
from neural import *
from neural_net_UCI_data import parse_line
from neural_net_UCI_data import normalize
from sklearn.model_selection import train_test_split

with open("abalone_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

train_data_split, test_data_split = train_test_split(training_data)

for line in training_data:
    print(line)

print(len(training_data))
print(len(train_data_split))
print(len(test_data_split))

#td = normalize(training_data)
train_data_norm = normalize(train_data_split)
test_data_norm = normalize(test_data_split)

#for line in test_data_norm:
   # print(line)
#for line in td:
   # print(line)

nn = NeuralNet(13, 3, 1)
nn.train(train_data_norm, iters=10000, print_interval=1000, learning_rate=0.5)

for i in nn.test_with_expected(test_data_norm):
    print(f"desired: {i[1]}, actual: {i[2]}")   