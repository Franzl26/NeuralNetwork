from net import *
from funktionen import *

if __name__ == '__main__':
    print(round(sigmoid(0.904), 3) * -1.205)
    print(round(sigmoid(1.898), 3) * 1.094)
    print(round(sigmoid(0.904), 3) * -1.205 + round(sigmoid(1.898), 3) * 1.094)
    print(sigmoid(round(sigmoid(0.904), 3) * -1.205 + round(sigmoid(1.898), 3) * 1.094))

for i in range(-100, 100):
    print(f"{i}: {sprung(i / 100)}")
