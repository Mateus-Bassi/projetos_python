import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def carregar_dados():
    df = pd.read_csv("uploads/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    colunas_atributos = ['Age', 'Sex', 'BMI', 'CholCheck']
    X = df[colunas_atributos]
    y = df['Diabetes_binary']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y

def treinar_e_avaliar(classificador, parametros, X_train, y_train, X_test, y_test):
    if classificador == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=parametros['n_neighbors'])
    elif classificador == 'SVM':
        clf = SVC(kernel=parametros['kernel'], degree=parametros['degree'])
    elif classificador == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=parametros['hidden_layer_sizes'], max_iter=parametros['max_iter'])
    elif classificador == 'DT':
        clf = DecisionTreeClassifier(max_depth=parametros['max_depth'])
    elif classificador == 'RF':
        clf = RandomForestClassifier(n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'])
    else:
        raise ValueError("Classificador não encontrado.")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    # Salvando a matriz de confusão como imagem
    nome_imagem = f'confusion_matrix_{classificador}.png'
    caminho_imagem = f'static/{nome_imagem}'
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    plt.savefig(caminho_imagem)
    plt.close(fig)  # Fecha a figura para liberar memória

    return nome_imagem



