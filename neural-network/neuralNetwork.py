import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

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

if __name__ == "__main__":
    # --- Carregamento e Preparação dos Dados ---

    # Carregar o dataset
    file_path = 'treino_sinais_vitais_com_label.csv'
    df = pd.read_csv(file_path)

    # Definir as features (X) e o alvo (y)
    # 'p_sist' e 'p_diast' são excluídas, assim como 'id' e 'classe'.
    features = ['qpa', 'pulso', 'resp', 'gravidade']
    target = 'classe'

    X = df[features].values

    # a) Ajustar classes: Nossas classes são [1, 2, 3, 4].
    #    Para a matemática de arrays (e one-hot encoding), é muito melhor que sejam [0, 1, 2, 3].
    #    Vamos subtrair 1.
    y = df[target].values.astype(int) - 1

    # --- Pré-processamento ---


    # b) One-Hot Encoding para o alvo (y)
    #    A rede terá 4 neurônios de saída, e o formato one-hot é necessário para a cross-entropy.
    #    Ex: classe 2 (agora 1) -> [0, 1, 0, 0]
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
    # y_one_hot terá a forma (1500, 4)

    # c) Padronização das Features (X)
    #    Redes neurais convergem muito melhor quando as features de entrada estão na mesma escala.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # d) Divisão em Treino e Validação
    #    Vamos usar 80% para treino e 20% para validar o modelo
    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_one_hot, test_size=0.2, random_state=42)

    # Para a avaliação final, também precisaremos dos rótulos de validação no formato original (não one-hot)
    y_val_indices = np.argmax(y_val, axis=1)

    # --- Treinamento e Avaliação do Modelo ---

    # Definir hiperparâmetros
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    input_size = x_train.shape[1]    # 4 features
    hidden_size = 64                 # 64 neurônios na camada escondida
    output_size = y_train.shape[1]   # 4 classes
    epochs = 2000
    learning_rate = 0.05

    print("--- Iniciando Treinamento da Rede Neural 'do Zero' ---")

    # Instanciar a rede
    nn = NeuralNetworkNumpy(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Treinar a rede
    loss_history = nn.fit(x_train, y_train, epochs=epochs, learning_rate=learning_rate, print_loss_every=200)

    print("--- Treinamento Concluído ---")

    # --- Avaliação ---

    print("\n--- Avaliação no Conjunto de Validação ---")
    # Fazer previsões no conjunto de validação
    y_pred = nn.predict(x_val)

    # y_pred está em índices [0, 1, 2, 3]
    # y_val_indices (que criamos lá em cima) também está [0, 1, 2, 3]

    # Calcular a acurácia
    accuracy = accuracy_score(y_val_indices, y_pred)
    print(f"Acurácia no conjunto de validação: {accuracy * 100:.2f}%")

    # Imprimir relatório de classificação detalhado
    # (lembrando que 0=classe 1, 1=classe 2, etc.)
    print("\nRelatório de Classificação Detalhado:")
    # Usamos os nomes das classes originais para clareza
    print(classification_report(y_val_indices, y_pred, target_names=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']))
