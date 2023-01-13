from net import *
import funktionen

train = [((0, 0), [0]), ((0, 1), [0]), ((1, 0), [1]), ((1, 1), [0])]

net = get_net([2, 1], (lambda d: d))

if __name__ == '__main__':
    # print(net["layers"])
    # print(net["anz"])
    # print(net["w"])
    # print(net["b"])

    train_hebb(net, train, True)
    print(use_net(net, [(0, 0), (0, 1), (1, 0), (1, 1)]))

    net = get_net([2, 1], (lambda d: d))
    train = [((0, 0), [1]), ((0, 1), [0]), ((1, 0), [0]), ((1, 1), [1])]
    train_hebb(net, train, True)
    print(use_net(net, [(0, 0), (0, 1), (1, 0), (1, 1)]))

