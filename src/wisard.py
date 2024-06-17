import numpy as np
from bitarray import bitarray
from utils import SEED
from copy import deepcopy

def bitarray_to_np(arr:bitarray):
    l = []
    for bit in arr: l.append(bit)
    return np.array(l)

def np_to_bitarray(arr:np.ndarray)->bitarray:
    string = ""
    for bit in arr:string += str(bit)
    return bitarray(string)

def bitarray_shuffle(arr:bitarray, random_state:int=SEED)->bitarray:
    np_arr = bitarray_to_np(arr)
    np.random.seed(seed=random_state)
    np.random.shuffle(np_arr)
    return np_to_bitarray(np_arr)

def binarize_list(arr:list)->list:
    max_elem = max(arr)
    return [1 if elem==max_elem else 0 for elem in arr]

def pred_deadlock(arr:list)->None:
    # Deactivate every bit except the last
    while arr.count(1) > 1:
        arr[arr.index(1)]=0


class Discriminator:
    def __init__(self, ft_size:int, rams:int=4):
        # Check if feature size is divisible by rams
        if ft_size % rams != 0: raise Exception(f"ft_size({ft_size}) is not divisible by rams({rams})")
        self.dicts = [{} for _ in range(rams)]
        self.ft_size = ft_size
        self.n_rams = rams

    def train(self, X:list[list[bitarray]], y:list[int]):
        # print(X)
        for arr_list, label in zip(X, y):
            if label==0:continue
            for arr, dict_ in zip(arr_list, self.dicts):
                dict_[arr.to01()] = label
    
    def predict(self, X:list[bitarray])->list[int]:
        pred = 0
        # print(X)
        for arr, dict_ in zip(X, self.dicts):
            if arr.to01() in dict_.keys(): pred += 1
        return pred

class WiSARD:
    
    def __init__(self, ft_size:int, num_classes:int, rams:int=4):
        self.ft_size = ft_size
        self.num_classes = num_classes
        self.rams = rams
        self.discriminators:list[Discriminator] = [Discriminator(ft_size=ft_size, rams=rams) for _ in range(num_classes)]
    
    def data_for_rams(self, X:list[bitarray])->list[list[bitarray]]:
        ft_size = self.ft_size
        rams = self.rams

        pieces = []
        for arr in X:
            arr = bitarray_shuffle(arr)
            arr_pieces = []
            for i in range(rams):
                arr_pieces.append(
                    arr[int(ft_size/rams)*i:int(ft_size/rams)*i+int(ft_size/rams)]
                )
            pieces.append(arr_pieces)
        return pieces

    def train(self, X:list[bitarray], y:np.ndarray)->None:
        if type(y)!= np.ndarray:y = np.array(y)
        if y.shape[1] != self.num_classes:
            raise Exception(f"Number of columns in y ({y.shape[1]}) is different to number of classes ({self.num_classes})")
        if type(y)==list:y=np.array(y)
        X_parted = self.data_for_rams(X)
        for idx, zip_info in enumerate(zip(self.discriminators, y.transpose())):
            discriminator, labels = zip_info
            # print("Training discriminator: ", idx)
            discriminator.train(X_parted, labels)
    
    def predict(self, X:list[bitarray])->list[list[int]]:
        # If WiSARD has been trained on incompatible data 
        preds = []
        X_parted = self.data_for_rams(X)
        for data in X_parted: 
            disc_res = []
            for idx, discriminator in enumerate(self.discriminators):
                # print("Predicting with discriminator", idx)
                disc_res.append(
                    discriminator.predict(data))
            disc_res = binarize_list(disc_res)
            pred_deadlock(disc_res)
            preds.append(disc_res)
        
        return preds


if __name__ == "__main__":
  
    # TODO: Test WiSARD thoroughly

    ft_size = 8
    rams = 2
    
    X = [
        bitarray("10101010"),
        bitarray("10101010"),
        bitarray("01010101"),
        bitarray("00110011"),
    ]

    y = [
        [1,0],
        [1,0],
        [0,1],
        [1,0],
    ]
    
    wisard = WiSARD(ft_size=ft_size, rams=rams, num_classes=2)

    wisard.train(X, y)

    preds = wisard.predict([
        bitarray("10101010"),
        bitarray("10101111"),
        bitarray("11101110"),
        bitarray("01010101"),
    ])
    # print(preds)


        