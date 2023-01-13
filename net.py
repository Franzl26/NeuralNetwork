import random


def get_net(anz_neurons, func, func_derivation=None, weight_random=False):
    w = {}
    b = {}
    f = {}
    f_der = {}
    for layer in range(0, len(anz_neurons) - 1):
        for start in range(1, anz_neurons[layer] + 1):
            for end in range(1, anz_neurons[layer + 1] + 1):
                s = str(layer + 1) + str(start) + str(end)
                if weight_random:
                    w[s] = random.random() * 1 - 0.5
                else:
                    w[s] = 0
    for layer in range(1, len(anz_neurons)):
        for neu in range(1, anz_neurons[layer] + 1):
            s = str(layer) + str(neu)
            b[s] = 0
            f[s] = func
            f_der[s] = func_derivation

    if func_derivation is None:
        f_der = None

    dic = {
        "f": f,
        "f_der": f_der,
        "layers": len(anz_neurons),
        "anz": anz_neurons,
        "w": w,  # weights
        "b": b  # bias
    }
    return dic


def net_set_weight(net, weights):
    w = net["w"]
    for layer in range(1, net["layers"]):
        for anz_cur in range(1, net["anz"][layer - 1] + 1):
            for anz_next in range(1, net["anz"][layer] + 1):
                s = f"{layer}{anz_cur}{anz_next}"
                w[s] = weights[layer - 1][anz_cur - 1][anz_next - 1]


def net_set_bias(net, biases):
    b = net["b"]
    for layer in range(1, net["layers"]):
        for anz_cur in range(1, net["anz"][layer] + 1):
            s = f"{layer}{anz_cur}"
            b[s] = biases[layer - 1][anz_cur - 1]


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
            if ausgabe:
                print("y = " + str(y) + "  ", end="")
            for anz_in in range(1, net["anz"][0] + 1):
                s = "1" + str(anz_in) + str(anz_out)
                w[s] += (sample[1][anz_out - 1] - y) * sample[0][anz_in - 1]
            b[s_bias] -= (sample[1][anz_out - 1] - y)
        count += 1
    if ausgabe:
        print(str(count) + " " + str(w) + " " + str(b))


def train_backpropagation(net, data, lernrate, fehlerformel, fehlerformel_ableitung, ausgabe=False):
    if net["f_der"] is None:
        print("Keine Ableitung der Übertragungsfunktion vorhanden")
        raise ValueError

    w = net["w"]
    b = net["b"]
    count = 0
    for sample in data:
        if ausgabe:
            print(f"w {count}: {w} {b}")

        # Vorwärts
        erg = list(sample[0])
        erg_neu = []
        neuron_in = {}
        neuron_out = {}
        for anz in range(1, net["anz"][0] + 1):
            s_neuron = f"{0}{anz}"
            neuron_out[s_neuron] = sample[0][anz - 1]
        for layer in range(1, net["layers"]):
            for anz_out in range(1, net["anz"][layer] + 1):
                summe = 0
                for anz_in in range(1, net["anz"][layer - 1] + 1):
                    s_neuron = f"{layer}{anz_in}{anz_out}"
                    summe += w[s_neuron] * erg[anz_in - 1]
                s_bias = f"{layer}{anz_out}"
                summe -= b[s_bias]
                neuron_in[s_bias] = summe
                calc = net["f"][f"{layer}{anz_out}"](summe)
                neuron_out[s_bias] = calc
                erg_neu.append(calc)
            erg = erg_neu
            erg_neu = []

        # print("in : " + str(neuron_in))
        # print("out: " + str(neuron_out))

        # Deltas ausrechnen
        f_der = net["f_der"]
        deltas = {}
        for layer in range(net["layers"] - 1, 0, -1):
            if layer == net["layers"] - 1:
                for anz in range(1, net["anz"][layer] + 1):
                    s_neuron = f"{layer}{anz}"
                    sig_der = f_der[s_neuron](neuron_in[s_neuron])
                    # print(sample[1][anz - 1])
                    # fehler = fehlerformel(neuron_out[s_neuron], sample[1][anz - 1])
                    fehler = fehlerformel_ableitung(neuron_out[s_neuron], sample[1][anz - 1])
                    # print("sig_der:" + str(sig_der))
                    # print("fehler :" + str(fehler))
                    # print("delta  :" + str(sig_der * fehler))
                    deltas[s_neuron] = sig_der * fehler
            else:
                for anz_cur in range(1, net["anz"][layer] + 1):
                    summe = 0
                    for anz_next in range(1, net["anz"][layer + 1] + 1):
                        s_neuron = f"{layer + 1}{anz_next}"
                        s_con = f"{layer + 1}{anz_cur}{anz_next}"
                        summe += deltas[s_neuron] * w[s_con]
                    s_neuron = f"{layer}{anz_cur}"
                    sig_der = f_der[s_neuron](neuron_in[s_neuron])
                    deltas[f"{layer}{anz_cur}"] = summe * sig_der

        # Gewichtsänderungen ausrechnen und anpassen
        delta_w = {}
        for layer in range(net["layers"] - 1, 0, -1):
            for anz in range(1, net["anz"][layer] + 1):
                s_neuron = f"{layer}{anz}"
                for anz_last in range(1, net["anz"][layer - 1] + 1):
                    s_con = f"{layer}{anz_last}{anz}"
                    s_neuron_last = f"{layer - 1}{anz_last}"
                    delta = - lernrate * deltas[s_neuron] * neuron_out[s_neuron_last]
                    delta_w[s_con] = delta
                    w[s_con] += delta

        if ausgabe:
            print(f"ẟ  : {deltas}")
            print(f"Δw : {delta_w}")
        count += 1
    if ausgabe:
        print(f"w {count}: {w} {b}")


"""
def train_backpropagation_v2(net, data, lernrate, fehlerformel, ausgabe=False):
    if net["f_der"] is None:
        print("Keine Ableitung der Übertragungsfunktion vorhanden")
        raise ValueError

    w = net["w"]
    b = net["b"]
    count = 0
    for sample in data:
        if ausgabe:
            print(f"{count} {w} {b}")

        # Vorwärts
        erg = list(sample[0])
        erg_neu = []
        neuron_in = {}
        neuron_out = {}
        for anz in range(1, net["anz"][0] + 1):
            s_neuron = f"{0}{anz}"
            neuron_out[s_neuron] = sample[0][anz - 1]
        for layer in range(1, net["layers"]):
            for anz_out in range(1, net["anz"][layer] + 1):
                summe = 0
                for anz_in in range(1, net["anz"][layer - 1] + 1):
                    s_neuron = f"{layer}{anz_in}{anz_out}"
                    summe += w[s_neuron] * erg[anz_in - 1]
                s_bias = f"{layer}{anz_out}"
                summe -= b[s_bias]
                neuron_in[s_bias] = summe
                calc = net["f"][f"{layer}{anz_out}"](summe)
                erg_neu.append(calc)
                neuron_out[s_bias] = calc
            erg = erg_neu
            erg_neu = []

        # Deltas ausrechnen
        f_der = net["f_der"]
        deltas = {}
        for layer in range(net["layers"] - 1, 0, -1):
            if layer == net["layers"] - 1:
                for anz in range(1, net["anz"][layer] + 1):
                    s_neuron = f"{layer}{anz}"
                    sig_der = f_der[s_neuron](neuron_in[s_neuron])
                    fehler = fehlerformel(neuron_out[s_neuron], sample[1][anz - 1])
                    deltas[s_neuron] = sig_der * fehler
            else:
                for anz_cur in range(1, net["anz"][layer] + 1):
                    summe = 0
                    for anz_next in range(1, net["anz"][layer + 1] + 1):
                        s_neuron = f"{layer + 1}{anz_next}"
                        s_con = f"{layer + 1}{anz_cur}{anz_next}"
                        summe += deltas[s_neuron] * w[s_con]
                    s_neuron = f"{layer}{anz_cur}"
                    sig_der = f_der[s_neuron](neuron_in[s_neuron])
                    deltas[f"{layer}{anz_cur}"] = summe * sig_der

        # Gewichtsänderungen ausrechnen und anpassen
        delta_w = {}
        for layer in range(net["layers"] - 1, 0, -1):
            for anz in range(1, net["anz"][layer] + 1):
                s_neuron = f"{layer}{anz}"
                for anz_last in range(1, net["anz"][layer - 1] + 1):
                    s_con = f"{layer}{anz_last}{anz}"
                    s_neuron_last = f"{layer - 1}{anz_last}"
                    delta = - lernrate * deltas[s_neuron] * neuron_out[s_neuron_last]
                    delta_w[s_con] = delta
                    w[s_con] += delta

        if ausgabe:
            print(f"deltas: {deltas}")
            print(f"deltas der Gewichte: {delta_w}")
        count += 1
    if ausgabe:
        print(f"{count} {w} {b}")
"""
