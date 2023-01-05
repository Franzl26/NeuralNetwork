from net import *

train = [((0, 0), [0]), ((1, 0), [1]), ((0, 1), [1])]

net = get_net([2, 1], (lambda d: 1 if d >= 0 else 0))

if __name__ == '__main__':
    train_perceptron(net, train, True)
    print(use_net(net, get_test_from_train(train)))
