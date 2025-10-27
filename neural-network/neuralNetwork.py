import numpy as np

# --- mplementação da Rede Neural "do Zero" (com NumPy) ---

class NeuralNetworkNumpy:
    """
    Uma rede neural simples de uma camada escondida, implementada
    usando apenas NumPy.

    Arquitetura:
    Entrada -> Camada Escondida (com ativação ReLU) -> Camada de Saída (com ativação Softmax)
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização de pesos e vieses (Xavier/Glorot para ReLU)
        # Camada 1 (Entrada -> Escondida)
        np.random.seed(42) # Para reprodutibilidade
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # Camada 2 (Escondida -> Saída)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        # Cache para guardar valores do forward prop para usar no backprop
        self.cache = {}

    # --- Funções de Ativação e Custo ---

    def _relu(self, Z):
        # Função de ativação ReLU (Rectified Linear Unit)
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        # Derivada da ReLU
        return (Z > 0) * 1

    def _softmax(self, Z):
        # Função de ativação Softmax (para classificação multiclasse)
        # Adicionamos -np.max(Z) para estabilidade numérica, evitando overflow com 'exp'
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _compute_loss(self, y_true, y_pred_probs):
        # Função de Custo: Cross-Entropy Categórica
        # y_true está no formato one-hot
        n_samples = len(y_true)

        # Adicionamos 1e-9 para evitar log(0)
        # Selecionamos apenas as probabilidades da classe correta
        correct_log_probs = -np.log(y_pred_probs[np.arange(n_samples), np.argmax(y_true, axis=1)] + 1e-9)
        loss = np.sum(correct_log_probs) / n_samples
        return loss

    # --- Forward Propagation (Passagem para Frente) ---

    def forward(self, X):
        # Camada 1
        Z1 = X.dot(self.W1) + self.b1
        A1 = self._relu(Z1)

        # Camada 2
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self._softmax(Z2) # Probabilidades de saída

        # Guardar valores para o backpropagation
        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    # --- Backward Propagation (Retropropagação) ---

    def backward(self, X, y_true):
        n_samples = X.shape[0]

        # Recuperar valores do cache
        Z1, A1, Z2, A2 = self.cache["Z1"], self.cache["A1"], self.cache["Z2"], self.cache["A2"]

        # --- Cálculo dos Gradientes ---

        # 1. Gradiente da Camada de Saída (Softmax + Cross-Entropy)
        # O gradiente dL/dZ2 é simplesmente (A2 - y_true)
        # (y_true está em formato one-hot)
        dZ2 = A2 - y_true

        # 2. Gradientes para W2 e b2
        dW2 = (1 / n_samples) * A1.T.dot(dZ2)
        db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)

        # 3. Gradiente da Camada Escondida
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self._relu_derivative(Z1) # Aplicando derivada da ReLU

        # 4. Gradientes para W1 e b1
        dW1 = (1 / n_samples) * X.T.dot(dZ1)
        db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)

        # Retornar dicionário de gradientes
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    # --- Treinamento e Predição ---

    def fit(self, X, y, epochs, learning_rate, print_loss_every=100):
        """ Loop de treinamento principal """
        history = []
        for i in range(epochs + 1):
            # 1. Forward pass
            y_pred_probs = self.forward(X)

            # 2. Calcular Custo
            loss = self._compute_loss(y, y_pred_probs)
            history.append(loss)

            # 3. Backward pass (calcular gradientes)
            grads = self.backward(X, y)

            # 4. Atualizar parâmetros (Gradiente Descendente)
            self.W1 -= learning_rate * grads["dW1"]
            self.b1 -= learning_rate * grads["db1"]
            self.W2 -= learning_rate * grads["dW2"]
            self.b2 -= learning_rate * grads["db2"]

            if i % print_loss_every == 0:
                print(f"Época {i}/{epochs} - Custo: {loss:.4f}")

        return history

    def predict(self, X):
        """ Faz previsões (retorna a classe, não as probabilidades) """
        # Executa o forward pass para obter as probabilidades
        probs = self.forward(X)
        # Retorna o índice (classe) com a maior probabilidade
        return np.argmax(probs, axis=1)

