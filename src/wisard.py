from bitarray import bitarray

class Discriminator:
    def __init__(self, ft_size:int, rams:int=4):
        pass

    def train(self, X:list[bitarray], y:list[bitarray]):
        pass
    
    def predict(self, X:list[bitarray]):
        pass

class WiSARD:
    
    def __init__(self, ft_size:int, num_classes:int, rams_per_class:int=4):
        self.ft_size = ft_size
    
    def train(self, X:list[bitarray], y:list[bitarray]):
        pass
    
    def predict(self, X:list[bitarray]):
        # If WiSARD has been trained on incompatible data 
        for data in X: 
            if len(data) != self.ft_size: 
                raise Exception(f"This classifier has been trained with instances of size {self.ft_size} this instance has size {len(data)}")
        # 
        
        