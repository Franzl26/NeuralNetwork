def get_net(anz_neurons, func):
    w = {}
    b = {}
    f = {}
    for layer in range(0, len(anz_neurons) - 1):
        for start in range(1, anz_neurons[layer] + 1):
            for end in range(1, anz_neurons[layer + 1] + 1):
                s = str(layer + 1) + str(start) + str(end)
                w[s] = 0
    for layer in range(1, len(anz_neurons)):
        for neu in range(1, anz_neurons[layer] + 1):
            s = str(layer) + str(neu)
            b[s] = 0
            f[s] = func

    dic = {
        "f": f,
        "layers": len(anz_neurons),
        "anz": anz_neurons,
        "w": w,  # weights
        "b": b  # bias
    }
    return dic


def get_test_from_train(data):
    erg = []
    for sample in data:
        erg.append(sample[0])
    return erg


def use_net(net, data):
    w = net["w"]
    b = net["b"]
    erg_ges = []
    for sample in data:
        erg = list(sample)
        erg_neu = []
        for layer in range(1, net["layers"]):
            for anz_out in range(1, net["anz"][layer] + 1):
                summe = 0
                for anz_in in range(1, net["anz"][layer - 1] + 1):
                    s = str(layer) + str(anz_in) + str(anz_out)
                    summe += w[s] * erg[anz_in - 1]
                summe -= b[str(layer) + str(anz_out)]
                erg_neu.append(net["f"][str(layer) + str(anz_out)](summe))
            erg = erg_neu
            erg_neu = []
        erg_ges.append(erg)
    return erg_ges


def train_hebb(net, data, ausgabe=False):
    if net["layers"] != 2:
        print("netz muss 2 layer haben")
        raise ValueError

    w = net["w"]
    count = 0
    for sample in data:
        if ausgabe:
            print(str(count) + " " + str(w))
        for anz_out in range(1, net["anz"][1] + 1):
            for anz_in in range(1, net["anz"][0] + 1):
                s = "1" + str(anz_in) + str(anz_out)
                w[s] += sample[1][anz_out - 1] * sample[0][anz_in - 1]
                w[s] = net["f"]["11"](w[s])
        count += 1
    if ausgabe:
        print(str(count) + " " + str(w))


def train_perceptron(net, data, ausgabe=False):
    if net["layers"] != 2:
        print("netz muss 2 layer haben")
        raise ValueError

    w = net["w"]
    b = net["b"]
    count = 0
    for sample in data:
        if ausgabe:
            print(str(count) + " " + str(w) + " " + str(b))
        for anz_out in range(1, net["anz"][1] + 1):
            s_bias = "1" + str(anz_out)
            a = 0
            for anz_in in range(1, net["anz"][0] + 1):
                s = "1" + str(anz_in) + str(anz_out)
                a += w[s] * sample[0][anz_in - 1]
            a -= b[s_bias]
            y = net["f"]["11"](a)
            print("y = " + str(y) + "  ", end="")
            for anz_in in range(1, net["anz"][0] + 1):
                s = "1" + str(anz_in) + str(anz_out)
                w[s] += (sample[1][anz_out - 1] - y) * sample[0][anz_in - 1]
            b[s_bias] -= (sample[1][anz_out - 1] - y)
        count += 1
    if ausgabe:
        print(str(count) + " " + str(w) + " " + str(b))
