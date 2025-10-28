import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

from neuralNetwork import NeuralNetworkNumpy

PATH = Path(__file__).resolve().parent
DATASET_PATH =  PATH.parent / "datasets" / "treino_sinais_vitais_com_label.csv"

if __name__ == "__main__":
    # --- Carregamento e Preparação dos Dados ---

    with open(DATASET_PATH, "r") as f:
        first_line = f.readline()
    
    if "id" not in first_line:
        with open(DATASET_PATH, "r+") as f:
            conteudo = f.read()
            f.seek(0)  # volta pro início do arquivo
            nome_colunas = "id,p_sist,p_diast,qpa,pulso,resp,gravidade,classe\n"
            f.write(nome_colunas + conteudo)
            

    # Carregar o dataset
    df = pd.read_csv(DATASET_PATH)

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

