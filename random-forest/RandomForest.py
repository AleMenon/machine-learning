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
        self.forest = []

    # Treina a floresta aleatória nos dados fornecidos.
    def fit(self, data, labels):
        self.forest = []
        for _ in range(self.n_trees):
            # Inicializa uma nova árvore de decisão
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            # Gera um subconjunto de dados usando o método bootstrap
            data_sample, labels_sample = self._generate_bootstrap(data, labels)
            # Treina a árvore no subconjunto gerado
            tree.fit(data_sample, labels_sample)
            # Adiciona a árvore treinada à floresta
            self.forest.append(tree)

    # Gera um subconjunto de dados usando o método bootstrap.
    def _generate_bootstrap(self, data, labels):
        n_samples = data.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return data[indices], labels[indices]

    # Encontra o rótulo mais comum em um conjunto de rótulos.
    def _most_frequent_label(self, labels):
        counter = Counter(labels)
        most_frequent = counter.most_common(1)[0][0]
        return most_frequent

    # Faz previsões para os dados fornecidos usando a floresta aleatória.
    def predict(self, data):
        # Obtém as previsões de todas as árvores para cada amostra
        all_predictions = np.array([tree.predict(data) for tree in self.forest])
        # Transforma as previsões para que cada linha corresponda a uma amostra
        sample_predictions = np.swapaxes(all_predictions, 0, 1)
        # Para cada amostra, encontra o rótulo mais comum entre as árvores
        final_predictions = np.array([self._most_frequent_label(pred) for pred in sample_predictions])
        return final_predictions