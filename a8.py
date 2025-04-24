from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

xornone = NeuralNet(2, 2, 1)
xornone.train(xor_training_data, learning_rate=.7, iters=10000, print_interval=1000)
print(xornone.test_with_expected(xor_training_data))

xorntwo = NeuralNet(2, 8, 1)
xorntwo.train(xor_training_data, iters=10000, print_interval=500)
print(xorntwo.test_with_expected(xor_training_data))

xornthree = NeuralNet(2, 1, 1)
xornthree.train(xor_training_data, learning_rate=.7, iters=10000, print_interval=1000)
print(xornthree.test_with_expected(xor_training_data))

xor_training_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]), ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])]

xorn = NeuralNet(5, 0 , 1)
xorn.train(xor_training_data, iters=10000, print_interval=1000)
print(xorn.test_with_expected(xor_training_data))
print(xorn.test([1.0, 1.0, 0.2, 0.6, 0.5]))
print(xorn.test([0.2, 0.1, 0.6, 0.9, 0.6]))
print(xorn.test([0.5, 0.2, 0.9, 1.0, 0.1]))
print(xorn.test([1.0, 1.0, 0.1, 1.0, 0.1]))
print(xorn.test([0.1, 0.2, 0.2, 0.4, 0.5]))