from net import *
import funktionen
import fehlerberechnung

train = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
# train = [[[0, 0], [0]], [[1, 0], [1]], [[0, 1], [1]], [[1, 1], [0]]]

net = get_net([2, 2, 1], funktionen.sigmoid, funktionen.sigmoid_derivation, weight_random=True)
net_set_weight(net, [[[0.5, -0.5], [-0.5, 0.5]], [[1], [1]]])
net_set_bias(net, [[1 / 3, 1 / 3], [0]])

print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))
print(net["w"])

for i in range(100000):
    train_backpropagation(net, train, 0.001, fehlerberechnung.squared_error)

print(net["w"])
print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))
