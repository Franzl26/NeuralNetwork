def squared_error(ist, soll):
    return 0.5 * (soll - ist) ** 2


def squared_error_ableitung(ist, soll):
    return ist - soll


def total_error(ist, soll):
    return 0.5 * abs(soll - ist)


def total_error_ableitung(ist, soll):
    raise EnvironmentError


def bias_error(ist, soll):
    return 0.5 * (soll - ist)


def bias_error_ableitung(ist, soll):
    return -0.5
