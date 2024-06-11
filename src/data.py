# This will be used to import the dataset and preprocess it for classification, in the form of train and test

import pandas as pd
import os
from utils import get_workdir, bitarray_to_string, SEED
from binary_encoders import Thermometer, OneHot
from sklearn.model_selection import train_test_split
from bitarray import bitarray

def binarize_dataset(data, column_encoder):
    for column in column_encoder:
        data[column] = [column_encoder[column].bitarray_encode(elem) for elem in data[column]]
    train_mtx = []
    for idx, row in data.iterrows():
        line = bitarray()
        for item in row:
            line.extend(item)
        train_mtx.append(line)
    return train_mtx

def create_column_encoder(data:pd.DataFrame)->dict:
    column_encoder = {}
    data_desc = data.describe()

    for column in data.columns:
        if column in data_desc.columns: # this is a numerical column
            column_encoder[column] = Thermometer(min=data_desc[column]["min"],max=data_desc[column]["max"],num_bits=32)
        else: # this is a categorical column
            column_encoder[column] = OneHot(list(data[column].unique()))
    
    return column_encoder

def load_abalone(stratify:bool):
    data = pd.read_csv(f"{get_workdir()}/dataset/abalone/data.csv")

    # If stratification was not requested then return train test split as is
    if not stratify: train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=SEED)
    else:
        # Count how many exampes in each label
        unique_rings = data["rings"].unique()
        ring_counts = {ring:0 for ring in unique_rings}
        for ring in unique_rings:
            ring_count = len(data[data["rings"]==ring])
            ring_counts[ring] = ring_count
        
        # Create a mask of very under represented examples
        represented_rings = {ring:ring_counts[ring] for ring in ring_counts if ring_counts[ring]>=5}
        unrepresented_rings = {ring:ring_counts[ring] for ring in ring_counts if ring_counts[ring]<5}

        # Separate represented (datamass) and unrepresented data (sprinkle)
        datamass = data
        for ring in unrepresented_rings:
            datamass = datamass[datamass["rings"]!=ring]

        sprinkle = data
        for ring in represented_rings:
            sprinkle = sprinkle[sprinkle["rings"]!=ring]
        
        sprinkle = sprinkle.sample(frac=1, random_state=SEED)
        
        train, test = train_test_split(datamass, test_size=0.3, shuffle=True, stratify=datamass["rings"], random_state=SEED)

        train = pd.concat([train,sprinkle[int(len(sprinkle)/2):]])
        test = pd.concat([test,sprinkle[:int(len(sprinkle)/2)]])

    # Separate Target Column
    train_label, test_label = train["rings"], test["rings"]
    train_fts, test_fts = train.drop(columns=["rings"]), test.drop(columns=["rings"])

    # Encode columns
    column_encoder = create_column_encoder(pd.concat([train_fts, test_fts]))
    train_fts = binarize_dataset(train_fts, column_encoder)
    test_fts = binarize_dataset(test_fts, column_encoder)

    return (train_fts, train_label), (test_fts, test_label)

def load_internet():
    tgt = "Attack_type"
    data = pd.read_csv(f"{get_workdir()}/dataset/internet/data.csv")
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    # Encode columns
    column_encoder = create_column_encoder(pd.concat([train_fts, test_fts]))
    train_fts = binarize_dataset(train_fts, column_encoder)
    test_fts = binarize_dataset(test_fts, column_encoder)

    return (train_fts, train_label), (test_fts, test_label)

def load_soybean():
    tgt = "Cultivar"
    data = pd.read_csv(f"{get_workdir()}/dataset/soybean/data.csv")
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    # Encode columns
    column_encoder = create_column_encoder(pd.concat([train_fts, test_fts]))
    print("column encoder: ", column_encoder)
    train_fts = binarize_dataset(train_fts, column_encoder)
    test_fts = binarize_dataset(test_fts, column_encoder)

    return (train_fts, train_label), (test_fts, test_label)

def load_dataset(dataset:str)->list[pd.DataFrame]:
    if dataset == "abalone_stratified":
        return load_abalone(stratify=True)
    elif dataset == "abalone":
        return load_abalone(stratify=False)
    elif dataset == "internet":
        return load_internet()
    elif dataset == "soybean":
        return load_soybean()
    
if __name__ == "__main__":
    print("testing load_abalone() with stratification...")
    train, test = load_abalone(stratify=True)
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_abalone() without stratification...")
    train, test = load_abalone(stratify=False)
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_internet()...")
    train, test = load_internet()
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_soybean()...")
    train, test = load_soybean()
    train, _ = train
    test, _ = test
    print(len(train), len(test))
