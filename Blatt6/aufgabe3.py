from net import *

train = [[[1, 1, 0], [1, 0]], [[1, 0, 0], [0, 1]]]

net = get_net([3, 2], (lambda d: 1 if d >= 0 else 0))

if __name__ == '__main__':
    print(net["f"])
    train_perceptron(net, train, True)
    print(use_net(net, get_test_from_train(train)))
