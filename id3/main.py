import pandas as pd
import numpy as np
from ID3 import DecisionTreeClassifier

PATH = "./datasets/treino_sinais_vitais_com_label.csv"

if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/treino_sinais_vitais_com_label.csv")
    dataset = dataset.drop(columns=["id", "p_sist", "p_diast", "gravidade"])

    dataset["qpa"] = pd.cut(
        dataset["qpa"], 
        bins=[-10, -6, -2, 2, 6, 10],
        labels=["muito baixa", "baixa", "normal", "alta", "muito alta"],
        include_lowest=True
    )

    dataset["pulso"] = pd.cut(
        dataset["pulso"],
        bins=[0, 60, 100, 140, 200],
        labels=["muito baixo", "normal", "alto", "muito alto"],
        include_lowest=True
    )

    dataset["resp"] = pd.cut(
        dataset["resp"],
        bins=[0, 12, 20, 22],
        labels=["baixa", "normal", "alta"],
        include_lowest=True
    )
    X = np.array(dataset.drop("classe", axis=1).copy())
    y = np.array(dataset["classe"].copy())
    feature_names = list(dataset.keys())[:3]
    print(dataset.head)

    tree = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
    print("System entropy {:.4f}".format(tree.entropy))
    # run algorithm id3 to build a tree
    tree.id3()
    tree.printTree()
