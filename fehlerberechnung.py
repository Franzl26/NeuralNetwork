def squared_error(ist, soll):
    return 0.5 * (ist - soll) ** 2


def total_error(ist, soll):
    return 0.5 * abs(ist - soll)


def bias_error(ist, soll):
    return 0.5 * (ist - soll)
