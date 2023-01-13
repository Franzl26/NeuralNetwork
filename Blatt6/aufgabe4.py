from net import *
import funktionen
import fehlerberechnung

train = [[[1, 1], [0]]]

net = get_net([2, 2, 1], funktionen.sigmoid, funktionen.sigmoid_ableitung)
net_set_weight(net, [[[0.5, 0.9], [0.4, 1.0]], [[-1.2], [1.1]]])
# net_set_bias(net, [[-1, -2], [-3]])

print(net["w"])
print(use_net(net, [[1, 1]]))
train_backpropagation(net, train, 0.1, fehlerberechnung.squared_error, fehlerberechnung.squared_error_ableitung, True)
print(net["w"])
print(use_net(net, [[1, 1]]))

print("\n" + "-" * 50 + "\n")

net2 = get_net([2, 2, 1], funktionen.sigmoid, funktionen.sigmoid_ableitung)
net_set_weight(net2, [[[0.5, 0.9], [0.4, 1.0]], [[-1.2], [1.1]]])
net_set_bias(net2, [[0.1, 0.1], [0.1]])
print(net2["w"])
print(net2["b"])
print(use_net(net2, [[1, 1]]))
train_backpropagation_bias(net2, train, 0.1, fehlerberechnung.squared_error, fehlerberechnung.squared_error_ableitung,
                           True)
print(net2["w"])
print(net2["b"])
print(use_net(net2, [[1, 1]]))
