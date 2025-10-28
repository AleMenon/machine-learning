from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
import pandas as pd
import graphviz
from pathlib import Path

np.random.seed(1234)

PATH = Path(__file__).resolve().parent
DATASET_PATH = PATH.parent / "datasets" / "treino_sinais_vitais_com_label.csv"

def visualize_tree_structure():
    print("\n--- Decision Tree Structure ---")
    feature_names = features.columns.tolist() 
    display_tree(clf.root, feature_names)

    feature_names = features.columns.tolist()

    generate_tree_graph(clf, feature_names)

def add_edges_and_nodes(dot, node, feature_names, node_id=0):
    if node.is_leaf_node():
        dot.node(str(node_id), label=f"Value: {node.value}", shape="box")
        return node_id + 1

    feature_name = feature_names[node.feature]
    label = f"{feature_name} <= {node.threshold:.2f}"
    dot.node(str(node_id), label=label, shape="ellipse")

    left_id = node_id + 1
    next_id = add_edges_and_nodes(dot, node.left, feature_names, left_id)
    dot.edge(str(node_id), str(left_id), label="True")

    right_id = next_id
    final_id = add_edges_and_nodes(dot, node.right, feature_names, right_id)
    dot.edge(str(node_id), str(right_id), label="False")

    return final_id

def generate_tree_graph(tree, feature_names, filename="decision_tree_depth10"):
    dot = graphviz.Digraph(comment='Decision Tree', format='svg')

    dot.attr(size='150,150')
    dot.attr('node', fontsize='8')
    dot.attr('edge', fontsize='8')
    dot.attr(ranksep='1.0', nodesep='0.5')

    add_edges_and_nodes(dot, tree.root, feature_names)

    try:
        dot.render(filename, view=True)
        print(f"\nTree saved as '{filename}.svg'.")
    except Exception as e:
        print(f"\nError generating tree image: {e}")

def display_tree(node, feature_names, depth=0):
    indent = "  " * depth

    if node.is_leaf_node():
        print(f"{indent}Predict Value: {node.value}")
        return

    feature_name = feature_names[node.feature]
    print(f"{indent}Condition: {feature_name} <= {node.threshold:.2f}")

    print(f"{indent}--> True (Left):")
    display_tree(node.left, feature_names, depth + 1)

    print(f"{indent}--> False (Right):")
    display_tree(node.right, feature_names, depth + 1)

def calculate_accuracy(y_test, y_pred):
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    return np.sum(y_test == y_pred) / len(y_test)  # Retorna a proporção de acertos

if __name__ == "__main__":
    with open(DATASET_PATH, "r") as f:
        first_line = f.readline()

    if "id" not in first_line:
        with open(DATASET_PATH, "r+") as f:
            content = f.read()
            f.seek(0)
            column_names = "id,p_sist,p_diast,qpa,pulso,resp,gravidade,classe\n"
            f.write(column_names + content)

    # Carrega o dataset e remove as colunas desnecessárias
    dataset = pd.read_csv(DATASET_PATH)
    dataset = dataset.drop(columns=["id", "p_sist", "p_diast", "gravidade"])

    features = dataset.drop("classe", axis=1)
    labels = dataset["classe"]

    feature_names = features.columns.tolist()
    class_names = [str(c) for c in sorted(labels.unique())]

    # Divide o dataset em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=1234
    )

    # Cria e treina o modelo de árvore de decisão
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train.values, y_train.values)

    # Faz previsões no conjunto de teste
    predictions = clf.predict(X_test.values)

    acc = calculate_accuracy(y_test, predictions)
    print(acc)

    visualize_tree_structure()
