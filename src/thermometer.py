# Código adaptado de Leandro Santiago, disponível no repositório:
# https://github.com/leandro-santiago/bloomwisard/blob/master/encoding/thermometer.py

from bitarray import bitarray
import numpy as np
import math

class Thermometer:
    def __init__(self, min, max, num_bits):
        self.min = min
        self.max = max
        self.interval = max - min
        self.num_bits = num_bits

    def bitarray_encode(self, data):
        bits = bitarray(self.num_bits)
        bits.setall(0)

        bits_activated = int(math.ceil(((data - self.min)/ self.interval ) * self.num_bits))

        bits[0:bits_activated] = 1

        return bits
    
    def np_encode(self, data):
        #print data, self.min, self.max, self.interval
        if self.interval != 0.0:
            bits_activated = int(math.ceil(((float(data) - self.min)/ self.interval ) * self.num_bits))
            
            if bits_activated > self.num_bits:
                binarray = np.ones(self.num_bits, dtype=bool)
            else:
                binarray = np.zeros(self.num_bits, dtype=bool)

                for i in range(bits_activated):
                    binarray[i] = 1
        else:
            binarray = np.ones(self.num_bits, dtype=bool)

        return binarray
    
if __name__ == "__main__":
    numeric_feature = np.array([
        15,
        16,
        2,
        6,
        5,
    ])

    thermometer = Thermometer(min(numeric_feature), max(numeric_feature), 10)
    print("bitarray")
    for number in numeric_feature: print(thermometer.bitarray_encode(numeric_feature[0]))
    print("np")
    for number in numeric_feature: print(thermometer.np_encode(numeric_feature[0]))