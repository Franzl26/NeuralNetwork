from net import *
import funktionen
import fehlerberechnung

train = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
# train = [[[0, 0], [0]], [[1, 0], [1]], [[0, 1], [1]], [[1, 1], [0]]]

net = get_net([2, 2, 1], funktionen.sigmoid, funktionen.sigmoid_derivation, weight_random=True)
# net_set_weight(net, [[[0.5, 0.9], [0.4, 1.0]], [[-1.2], [1.1]]])
# net_set_bias(net, [[-1, -2], [-3]])

print(net["w"])
# print(net["b"])

print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))
for i in range(1):
    train_backpropagation(net, train, 0.001, fehlerberechnung.squared_error)

print(net["w"])
print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))