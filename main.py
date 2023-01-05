from net import *

train = [((0, 0), [0]), ((1, 0), [1]), ((0, 1), [1])]

net = get_net([2, 2, 1], (lambda d: 1 if d >= 0 else 0))
net["f"]["21"] = lambda x: x

if __name__ == '__main__':
    # train_perceptron(net, train, True)
    w = net["w"]
    w["111"] = 1 / 2
    w["121"] = -1 / 2
    w["112"] = -1 / 2
    w["122"] = 1 / 2
    w["211"] = 1
    w["221"] = 1
    net["b"]["11"] = 1/3
    net["b"]["12"] = 1/3
    print(use_net(net, [[0, 0], [0, 1], [1, 0], [1, 1]]))
