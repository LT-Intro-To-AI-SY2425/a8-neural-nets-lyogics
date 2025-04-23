from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

xorn = NeuralNet(2, 2, 1)
xorn.train(xor_training_data, learning_rate=.7, iters=10000, print_interval=1000)
print(xorn.test_with_expected(xor_training_data))

xorn = NeuralNet(2, 8, 1)
xorn.train(xor_training_data, iters=10000, print_interval=500)
print(xorn.test_with_expected(xor_training_data))

xorn = NeuralNet(2, 1, 1)
xorn.train(xor_training_data, learning_rate=.7, iters=10000, print_interval=1000)
print(xorn.test_with_expected(xor_training_data))