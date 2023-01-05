def squared_error(ist, soll):
    return (ist - soll) ** 2


def total_error(ist, soll):
    return abs(ist - soll)


def bias_error(ist, soll):
    return ist - soll
