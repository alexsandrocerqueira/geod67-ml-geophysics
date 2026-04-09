import numpy as np

def normalize_std_mean(X,std=True,mean=True):
    """
    Input: X: numpy array of shape (n_amostras, n_atributos)
    Output: X: numpy array of shape (n_amostras, n_atributos) normalizada
            mean: médias dos atributos
            std: desvio padrão dos atributos
    """
    X_norm = X.copy()
    mu, sigma = np.mean(X_norm, axis=0), np.std(X_norm, axis=0)
    
    if mean:
        X_norm = X_norm - np.mean(X_norm, axis=0)
    if std:
        X_norm = X_norm / np.std(X_norm, axis=0)
    return X_norm, mu, sigma
    