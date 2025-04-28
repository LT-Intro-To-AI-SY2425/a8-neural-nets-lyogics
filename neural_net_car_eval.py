from typing import Tuple
from neural import *
from neural_net_UCI_data import parse_line

with open("car_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]