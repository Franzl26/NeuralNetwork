import funktionen
from net import *

train = [((0, 0), [0]), ((1, 0), [1]), ((0, 1), [1])]

net = get_net([2, 2, 1], (lambda d: 1 if d >= 0 else 0))
net["f"]["21"] = lambda x: x

net2 = get_net([2, 2, 1], funktionen.sigmoid)

if __name__ == '__main__':
    # train_perceptron(net, train, True)
    net_set_weight(net2, [[[0.5, -0.5], [-0.5, 0.5]], [[1], [1]]])
    net_set_bias(net2, [[1 / 3, 1 / 3], [0]])
    w = net["w"]
    w["111"] = 1 / 2
    w["121"] = -1 / 2
    w["112"] = -1 / 2
    w["122"] = 1 / 2
    w["211"] = 1
    w["221"] = 1
    net["b"]["11"] = 1/3
    net["b"]["12"] = 1/3
    print(net["w"])
    print(net2["w"])
    print(net["b"])
    print(net2["b"])
    print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))
    print(use_net(net2, [[0, 0], [0, 1], [1, 0], [1, 1]]))
