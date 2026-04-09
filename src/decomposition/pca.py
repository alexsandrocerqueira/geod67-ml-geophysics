import numpy as np
from scipy.linalg import eigh

def PCA(X, n_components=None):
    """
    Input - X: matriz de dados normalizada (n_samples, n_features)
        - n_components: número de componentes principais a serem retornados
        
    Output - matriz de dados transformada (n_samples, n_components)
    """
    # Centralizando os dados
    X_centered = X - np.mean(X, axis=0)

    # Matriz de covariância
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Autovalores e autovetores
    eigenvalues, eigenvectors = eigh(cov_matrix)

    # Ordenando em ordem decrescente
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Selecionando o número de componentes
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    # Projeção dos dados
    X_transformed = np.dot(X_centered, eigenvectors)

    return X_transformed,eigenvalues,eigenvectors

def PCA_inversa(X_transformed, eigenvectors):
    """
    Input - X_transformed: matriz de dados transformada (n_samples, n_components)
        - eigenvectors: matriz de autovetores (n_features, n_components)
        
    Output - matriz de dados reconstruída (n_samples, n_features)
    """
    return np.dot(X_transformed, eigenvectors.T)