import os
from bitarray import bitarray

def get_workdir()->str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

def save_results(dataset:str, metrics:dict, args:str)->None:
    output_path = f"{get_workdir()}/results/{dataset}_results.csv"
    acc = metrics["accuracy"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1"]

    if not os.path.exists(output_path):
        with open(output_path, mode="w") as file: 
            file.write("accuracy,precision,recall,f1,args\n")
    with open(output_path, mode="a") as file:
        file.write(f"{acc},{prec},{rec},{f1},{args}\n")

def bitarray_to_string(number:bitarray)->str:
    return str(number).split('\'')[1]

if __name__ == "__main__":
    
    # print("testing save_results()...")
    
    # metrics = {
    #     "accuracy":0.5,
    #     "precision":0.5,
    #     "recall":0.5,
    #     "f1":0.5
    # }

    # for _ in range(5): save_results("test", metrics, "{argumenrs}")

    # print("testing bitarray_to_string()...")

    # bin = bitarray([1,0,0,0,1])

    # print("binary number: ", bin)
    # print("stringified binary number: ", bitarray_to_string(bin))

    print("testing get_workdir()")

    print(get_workdir())