from net import *
import funktionen

train = [[[1, 1, 0], [1, 0]], [[1, 0, 0], [0, 1]]]

net = get_net([3, 2], funktionen.sprung)

if __name__ == '__main__':
    train_perceptron(net, train, False)
    for i in range(100):
        train_perceptron(net, train, False)
    print(use_net(net, get_test_from_train(train)))
    # print(net["w"])

    net = get_net([3, 2], funktionen.identitaet)
    train_perceptron(net, train, False)
    for i in range(100):
        train_perceptron(net, train, False)
    print(use_net(net, get_test_from_train(train)))

    net = get_net([3, 2], funktionen.sigmoid)
    train_perceptron(net, train, False)
    for i in range(100):
        train_perceptron(net, train, False)
    print(use_net(net, get_test_from_train(train)))
    # print(net["w"])
