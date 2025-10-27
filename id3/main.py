from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree # A árvore do Sklearn
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

np.random.seed(1234)

PATH = "./../datasets/treino_sinais_vitais_com_label.csv"

def tree_visual_representations():
    print("\n--- Estrutura da Árvore de Decisão ---")
    feature_names = X.columns.tolist() 
    print_tree(clf.root, feature_names)

    feature_names = X.columns.tolist() 

    plot_tree_graphviz(clf, feature_names)

def add_nodes_edges(dot, node, feature_names, node_id=0):
    
    if node.is_leaf_node():
        dot.node(str(node_id), label=f"Valor: {node.value}", shape="box")
        return node_id + 1
    
    feature_name = feature_names[node.feature]
    label = f"{feature_name} <= {node.threshold:.2f}"
    dot.node(str(node_id), label=label, shape="ellipse")
    
    left_child_id = node_id + 1
    
    next_id = add_nodes_edges(dot, node.left, feature_names, left_child_id)
    dot.edge(str(node_id), str(left_child_id), label="True")
    
    right_child_id = next_id
    
    final_id = add_nodes_edges(dot, node.right, feature_names, right_child_id)
    dot.edge(str(node_id), str(right_child_id), label="False")
    
    return final_id

def plot_tree_graphviz(tree, feature_names, filename="decision_tree_depth10"):
    
    dot = graphviz.Digraph(comment='Decision Tree', format='svg')
    
    dot.attr(size='150,150')
    
    dot.attr('node', fontsize='8')
    dot.attr('edge', fontsize='8')

    dot.attr(ranksep='1.0', nodesep='0.5')
    
    add_nodes_edges(dot, tree.root, feature_names)
    
    try:
        dot.render(filename, view=True)
        print(f"\nÁrvore salva como '{filename}.svg'.")
        
    except Exception as e:
        print(f"\nErro ao gerar a imagem da árvore: {e}")

def print_tree(node, feature_names, depth=0):
    
    indent = "  " * depth

    if node.is_leaf_node():
        print(f"{indent}Prever Valor: {node.value}")
        return

    feature_name = feature_names[node.feature]
    
    print(f"{indent}Condição: {feature_name} <= {node.threshold:.2f}")

    print(f"{indent}--> True (Esquerda):")
    print_tree(node.left, feature_names, depth + 1)

    print(f"{indent}--> False (Direita):")
    print_tree(node.right, feature_names, depth + 1)

def accuracy(y_test, y_pred):
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        return np.sum(y_test == y_pred) / len(y_test)

if __name__ == "__main__":

    dataset = pd.read_csv(PATH)
    dataset = dataset.drop(columns=["id", "p_sist", "p_diast", "gravidade"])

    X = dataset.drop("classe", axis=1)
    y = dataset["classe"]

    feature_names = X.columns.tolist()
    class_names = [str(c) for c in sorted(y.unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train.values, y_train.values)
    predictions = clf.predict(X_test.values)

    acc = accuracy(y_test, predictions)
    print(acc)

    tree_visual_representations()
