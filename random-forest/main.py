from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import graphviz
from RandomForest import RandomForest
from pathlib import Path

np.random.seed(1234)

PATH = Path(__file__).resolve().parent
DATASET_PATH = PATH.parent / "datasets" / "treino_sinais_vitais_com_label.csv"

# Função auxiliar para adicionar nós e arestas à visualização da árvore
def add_nodes_and_edges(dot, node, feature_names, node_id=0):
    if node.is_leaf_node():
        dot.node(str(node_id), label=f"Value: {node.value}", shape="box")
        return node_id + 1

    feature_name = feature_names[node.feature]
    label = f"{feature_name} <= {node.threshold:.2f}"
    dot.node(str(node_id), label=label, shape="ellipse")

    left_id = node_id + 1
    next_id = add_nodes_and_edges(dot, node.left, feature_names, left_id)
    dot.edge(str(node_id), str(left_id), label="True")

    right_id = next_id
    final_id = add_nodes_and_edges(dot, node.right, feature_names, right_id)
    dot.edge(str(node_id), str(right_id), label="False")

    return final_id

# Função para gerar uma visualização gráfica da árvore usando Graphviz
def visualize_tree(tree, feature_names, filename="decision_tree_depth10"):
    dot = graphviz.Digraph(comment='Decision Tree', format='svg')

    dot.attr(size='150,150')
    dot.attr('node', fontsize='8')
    dot.attr('edge', fontsize='8')
    dot.attr(ranksep='1.0', nodesep='0.5')

    add_nodes_and_edges(dot, tree.root, feature_names)

    try:
        dot.render(filename, view=True)
        print(f"\nTree saved as '{filename}.svg'.")
    except Exception as e:
        print(f"\nError generating tree image: {e}")

# Função para imprimir a árvore de decisão no console
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

# Função para calcular a acurácia do modelo
def calculate_accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

if __name__ == "__main__":
    with open(DATASET_PATH, "r") as f:
        first_line = f.readline()

    if "id" not in first_line:
        with open(DATASET_PATH, "r+") as f:
            content = f.read()
            f.seek(0)
            column_names = "id,p_sist,p_diast,qpa,pulso,resp,gravidade,classe\n"
            f.write(column_names + content)

    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=["id", "p_sist", "p_diast", "gravidade"])

    features = data.drop("classe", axis=1)
    labels = data["classe"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=1234
    )

    forest = RandomForest(n_trees=20)
    forest.fit(X_train.values, y_train.values)

    predictions = forest.predict(X_test.values)

    accuracy = calculate_accuracy(y_test, predictions)
    print(accuracy)

    feature_names = features.columns.tolist()

    if forest.forest:
        print("\n--- Tree Structure of the Forest ---")

        tree_to_view = forest.forest[0]

        display_tree(tree_to_view.root, feature_names)

        visualize_tree(tree_to_view, feature_names, filename="random_forest_tree_0")
    else:
        print("The forest was not trained or contains no trees.")
