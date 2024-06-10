
from bitarray import bitarray
import numpy as np
import math

# Código adaptado de Leandro Santiago, disponível no repositório:
# https://github.com/leandro-santiago/bloomwisard/blob/master/encoding/thermometer.py
class Thermometer:
    def __init__(self, min, max, num_bits):
        self.min = min
        self.max = max
        self.interval = max - min
        self.num_bits = num_bits

    def bitarray_encode(self, data):
        bits = bitarray(self.num_bits)
        bits.setall(0)
        if self.min == self.max: return bits
        try:
            bits_activated = int(math.ceil(((data - self.min)/ self.interval ) * self.num_bits))
        except:
            print(f"warning: arithmetic error check {data} in the dataset")
            return bits
        bits[0:bits_activated] = 1
        return bits

class OneHot:
    def __init__(self, classes:list[str]):
        self.classes = classes
        self.num_bits = len(classes)
    
    def bitarray_encode(self, data):
        bits = bitarray(self.num_bits)
        bits.setall(0)
        bits[self.classes.index(data)] = 1
        return bits


if __name__ == "__main__":
    print("testing Thermometer class...")
    numeric_feature = np.array([15,16,2,6,5,])
    thermometer = Thermometer(min(numeric_feature), max(numeric_feature), 10)
    for number in numeric_feature: print(thermometer.bitarray_encode(number))

    print("testing OneHot class")
    categorical_feature = ["a", "a", "b", "c", "b"]
    categorical_classes = ["a", "b", "c"]

    onehot = OneHot(categorical_classes)
    for category in categorical_feature: print(onehot.bitarray_encode(category))

