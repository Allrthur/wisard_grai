import os
import argparse
import data
import utils
from wisard import WiSARD
from binary_encoders import OneHot
import pandas as pd
from bitarray import bitarray
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def save_results(results:dict):pass

def bitarray_to_list(arr:bitarray):
    return [bit for bit in arr]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load Dataset
    train, test = data.load_dataset(args.dataset)
    X_train, y_train = train
    X_test, y_test = test

    # One Hot Encode Labels
    ohe = OneHot(classes=pd.concat([y_train, y_test]).unique().tolist())

    y_test = [ohe.bitarray_encode(elem) for elem in y_test.to_list()]
    y_test = [bitarray_to_list(elem) for elem in y_test]

    y_train = [ohe.bitarray_encode(elem) for elem in y_train.to_list()]
    y_train = [bitarray_to_list(elem) for elem in y_train]

    # Run WiSARD
    ft_len = len(X_train[0])
    num_classes = len(y_train[0])
    rams = 5
    if args.dataset=="abalone" or args.dataset=="abalone_stratified": rams = 5
    elif args.dataset=="internet": rams = 5
    elif args.dataset=="soybean": rams = 5
    elif args.dataset=="glass": rams = 5
    elif args.dataset=="hepatitis": rams = 6

    print("ft_len: ", ft_len)
    print("num_classes: ", num_classes)
    print("rams: ", rams)
    if ft_len % rams != 0: 
        print(f"Error: ft_len is not divisible by #rams chosen, try again with another")
        exit()

    wisard = WiSARD(ft_size=ft_len,
                    num_classes=num_classes,
                    rams=rams)
    
    wisard.train(X_train, y_train)
    preds = wisard.predict(X_test)

    acc = accuracy_score(preds, y_test)
    prec, rec, f1, _ = precision_recall_fscore_support(preds, y_test, average="macro")

    print("acc: ", acc)
    print("prec: ", prec)
    print("rec: ", rec)
    print("f1: ", f1)

    utils.save_results(
        dataset=args.dataset,
        args=str(args),
        metrics={
            "accuracy":acc,
            "precision":prec,
            "recall":rec,
            "f1":f1
        }
    )