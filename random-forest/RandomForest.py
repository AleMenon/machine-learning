from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

# Classe que implementa o algoritmo Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    # Treina a floresta aleatória nos dados fornecidos.
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Inicializa uma nova árvore de decisão
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            # Gera um subconjunto de dados usando o método bootstrap
            X_sample, y_sample = self._bootstrap_samples(X, y)
            # Treina a árvore no subconjunto gerado
            tree.fit(X_sample, y_sample)
            # Adiciona a árvore treinada à floresta
            self.trees.append(tree)

    # Gera um subconjunto de dados usando o método bootstrap.
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    # Encontra o rótulo mais comum em um conjunto de rótulos.
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # Faz previsões para os dados fornecidos usando a floresta aleatória.
    def predict(self, X):
        # Obtém as previsões de todas as árvores para cada amostra
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transforma as previsões para que cada linha corresponda a uma amostra
        tree_preds = np.swapaxes(predictions, 0, 1)
        # Para cada amostra, encontra o rótulo mais comum entre as árvores
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions