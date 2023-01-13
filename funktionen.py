import math


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def sigmoid_ableitung(value):
    sig = sigmoid(value)
    return sig * (1 - sig)


def sprung(value):
    return 1 if value >= 0 else 0


def sprung_ableitung(value):
    return 1


def identitaet(value):
    return value


def identitaet_ableitung(value):
    return 1
