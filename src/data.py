# This will be used to import the dataset and preprocess it for classification, in the form of train and test

import pandas as pd
import os
from utils import get_workdir, bitarray_to_string
from thermometer import Thermometer
from sklearn.model_selection import train_test_split


def load_abalone():
    data = pd.read_csv(f"{get_workdir()}/dataset/abalone/data.csv")
    # TODO: Create a train test split for abalone
    mask = data.groupby("rings").count().reset_index()
    data[data["rings"]]
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data["rings"])
    return train, test

def load_dataset(dataset:str)->list[pd.DataFrame]:
    if dataset == "abalone":
        return load_abalone()
    
if __name__ == "__main__":
    print(load_abalone())