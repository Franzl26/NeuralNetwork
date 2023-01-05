from net import *

train = [((3 / 5, 0, 4 / 5), (4, 3)), ((-4 / 5, 0, 3 / 5), (-1, 2))]
train.append(((0, 1, 0), (1, 1)))
train.append(((1, 0, 0), (0, 1)))

net = get_net([3, 2], (lambda d: d))

if __name__ == '__main__':
    # print(train[0])
    # print(train[0][0])
    # print(train[0][0][0])
    # print(net["layers"])
    # print(net["anz"])
    # print(net["w"])
    # print(net["b"])

    train_hebb(net, train, True)

    # print(net["w"])
    # print(net["b"])

    print(use_net(net, [(3 / 5, 0, 4 / 5), (-4 / 5, 0, 3 / 5), (0, 1, 0), (1, 0, 0)]))
    print(use_net(net, [(3 / 5, 0, 4 / 5), (-4 / 5, 0, 3 / 5), (0, 1, 0), (1, 0, 0)]))
