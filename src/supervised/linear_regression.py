import numpy as np
from scipy.linalg import inv

def solucao_pseudo_inversa(X,y,intercept=True):
    """
    Inputs: X: matriz de dados (n_amostras, n_atributos)
            y: vetor de respostas (n_amostras,)
    Output: vetor de coeficientes (n_atributos,)
    """
    if intercept==True:
        X = np.column_stack((X, np.ones(X.shape[0])))
    
    return inv(X.T @ X) @ X.T @ y

def predict(X,w,intercept=True):
    """
    Inputs: X: matriz de dados (n_amostras, n_atributos)
            w: vetor de coeficientes (n_atributos,)
    Output: vetor de respostas preditas (n_amostras,)
    """
    if intercept==True:
        X = np.column_stack((X, np.ones(X.shape[0])))
    
    return X @ w