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

# Função para gerar uma visualização gráfica da árvore usando Graphviz
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

# Função para imprimir a árvore de decisão no console
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

# Função para calcular a acurácia do modelo
def accuracy(y_true, y_pred):
    # Calcula a proporção de acertos
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

if __name__ == "__main__":
    with open(DATASET_PATH, "r") as f:
        first_line = f.readline()

    if "id" not in first_line:
        with open(DATASET_PATH, "r+") as f:
            conteudo = f.read()
            f.seek(0)
            nome_colunas = "id,p_sist,p_diast,qpa,pulso,resp,gravidade,classe\n"
            f.write(nome_colunas + conteudo)

    dataset = pd.read_csv(DATASET_PATH)
    dataset = dataset.drop(columns=["id", "p_sist", "p_diast", "gravidade"])

    X = dataset.drop("classe", axis=1)
    y = dataset["classe"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=20)
    clf.fit(X_train.values, y_train.values)

    predictions = clf.predict(X_test.values)

    acc = accuracy(y_test, predictions)
    print(acc)

    feature_names = X.columns.tolist()

    if clf.trees:
        print("\n--- Estrutura da Árvore da Floresta ---")

        arvore_para_ver = clf.trees[0]

        print_tree(arvore_para_ver.root, feature_names)

        plot_tree_graphviz(arvore_para_ver, feature_names, filename="random_forest_arvore_0")
    else:
        print("A floresta não foi treinada ou não contém árvores.")
