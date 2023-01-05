def mean_squared_error(y, y_quer):
    return (y - y_quer) * (y - y_quer)


def mean_total_error(y, y_quer):
    return abs(y - y_quer)


def mean_bias_error(y, y_quer):
    return y - y_quer
