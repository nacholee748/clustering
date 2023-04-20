import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA)
    
    Parameters:
    -----------
    n_components : int
        Number of principal components to compute.
        
    Attributes:
    -----------
    components_ : array-like, shape (n_components, n_features)
        Principal components.
        
    explained_variance_ : array-like, shape (n_components,)
        Amount of variance explained by each of the selected components.
        
    explained_variance_ratio_ : array-like, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        
    mean_ : array-like, shape (n_features,)
        Mean of each feature in the original data.

    Methods:
    --------
        fit_transform(matrix): Computes the PCA of a given matrix and returns its singular values, left and right singular vectors.
        fit_transform_truncated(matrix): Computes the truncated PCA of a given matrix and returns its singular values, left and right singular vectors.
            
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, matrix):
        """
        Fit the PCA model to the given data.
        
        Parameters:
        -----------
        matrix : array-like, shape (n_samples, n_features)
            Training data.
        """
        # Compute the mean of each feature
        self.mean_ = np.mean(matrix, axis=0)
        
        # Center the data by subtracting the mean
        matrix_centered = matrix - self.mean_
        
        # Compute the covariance matrix
        cov = np.cov(matrix_centered.T)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov)
        
        # Sort the eigenvectors and eigenvalues in descending order of eigenvalue
        eigvecs = eigvecs.T
        idxs = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[idxs]
        eigvals = eigvals[idxs]
        
        # Normalize the eigenvectors
        norm = np.sqrt(np.sum(np.power(eigvecs, 2), axis=1))
        eigvecs /= norm.reshape(-1, 1)
        
        # Store the first n_components eigenvectors
        if self.n_components is not None:
            self.components_ = eigvecs[:self.n_components]
            self.explained_variance_ = eigvals[:self.n_components]
        else:
            self.components_ = eigvecs
            self.explained_variance_ = eigvals
        
        # Compute the percentage of variance explained by each component
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
    def fit_transform(self, matrix):
        """
        Transform the given data into the principal component space.
        
        Parameters:
        -----------
        matrix : array-like, shape (n_samples, n_features)
            Data to transform.
        
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        # Compute the mean of each feature
        self.mean_ = np.mean(matrix, axis=0)

        # Center the data by subtracting the mean
        matrix_centered = matrix - self.mean_
        
        # Project the data onto the principal components
        matrix_transformed = np.dot(matrix_centered, self.components_.T)
        
        return matrix_transformed
