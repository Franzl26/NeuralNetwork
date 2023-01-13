def train_hopfield(vektoren):
    build = []
    for i in range(len(vektoren[0])):
        build.append(0)
    erg = []
    for i in range(len(vektoren[0])):
        erg.append(build)

    return erg
