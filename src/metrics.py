import numpy as np

def erro_quadratico_medio(y_true, y_pred):
    """
    Inputs: y_true: vetor de respostas verdadeiras (n_amostras,)
            y_pred: vetor de respostas preditas (n_amostras,)
    Output: valor do erro quadrático médio
    """
    return np.mean((y_true - y_pred)**2)

def erro_absoluto_medio(y_true, y_pred):
    """
    Inputs: y_true: vetor de respostas verdadeiras (n_amostras,)
            y_pred: vetor de respostas preditas (n_amostras,)
    Output: valor do erro absoluto médio
    """
    return np.mean(np.abs(y_true - y_pred))