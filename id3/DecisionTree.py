import numpy as np
from collections import Counter

# Classe que representa um nó na árvore de decisão
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Verifica se o nó é uma folha (não possui filhos)
    def is_leaf_node(self):
        return self.value is not None


# Classe que implementa a árvore de decisão
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    # Método para treinar a árvore de decisão
    def fit(self, data, labels):
        # Define o número de features a serem usadas
        self.n_features = data.shape[1] if not self.n_features else min(data.shape[1], self.n_features)
        # Constrói a árvore recursivamente
        self.root = self._build_tree(data, labels)

    # Método recursivo para construir a árvore
    def _build_tree(self, data, labels, depth=0):
        n_samples, n_feats = data.shape 
        n_classes = len(np.unique(labels))

        # Critérios de parada: profundidade máxima, nó puro ou poucas amostras
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(labels)
            return Node(value=leaf_value)

        # Seleciona aleatoriamente um subconjunto de features para avaliar
        feature_indices = np.random.choice(n_feats, self.n_features, replace=False)

        # Encontra a melhor divisão para o nó atual
        best_feature, best_thresh, best_gain = self._find_best_split(data, labels, feature_indices)
        if best_gain <= 0:
            leaf_value = self._most_common_label(labels)
            return Node(value=leaf_value)

        left_indices, right_indices = self._partition(data[:, best_feature], best_thresh)
        left = self._build_tree(data[left_indices, :], labels[left_indices], depth + 1)
        right = self._build_tree(data[right_indices, :], labels[right_indices], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    # Método para encontrar a melhor divisão para um nó
    def _find_best_split(self, data, labels, feature_indices):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feature_indices:
            feature_values = data[:, feat_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                gain = self._information_gain(labels, feature_values, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold, best_gain

    # Método para calcular o ganho de informação
    def _information_gain(self, labels, feature_values, threshold):
        parent_entropy = self._entropy(labels)

        left_indices, right_indices = self._partition(feature_values, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(labels)
        n_left, n_right = len(left_indices), len(right_indices)
        e_left, e_right = self._entropy(labels[left_indices]), self._entropy(labels[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    # Método para dividir os dados com base em um limiar
    def _partition(self, feature_values, threshold):
        left_indices = np.argwhere(feature_values <= threshold).flatten()
        right_indices = np.argwhere(feature_values > threshold).flatten()
        return left_indices, right_indices

    # Método para calcular a entropia
    def _entropy(self, labels):
        hist = np.bincount(labels)
        probabilities = hist / len(labels)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    # Método para encontrar o rótulo mais comum em um conjunto de dados
    def _most_common_label(self, labels):
        counter = Counter(labels)
        value = counter.most_common(1)[0][0]
        return value

    # Método para fazer previsões em novos dados
    def predict(self, data):
        return np.array([self._traverse_tree(sample, self.root) for sample in data])

    # Método recursivo para percorrer a árvore e fazer uma previsão
    def _traverse_tree(self, sample, node):
        if node.is_leaf_node():
            return node.value

        if sample[node.feature] <= node.threshold:
            return self._traverse_tree(sample, node.left)
        return self._traverse_tree(sample, node.right)
