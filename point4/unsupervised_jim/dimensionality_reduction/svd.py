import numpy as np

class SVD:
    """
    A class for performing Singular Value Decomposition (SVD) using NumPy.
    """
    
    def __init__(self, n_components=2):
        """
        Initialize the SVD class with a matrix to decompose.
        
        Args:
        - matrix (numpy.ndarray): The matrix to perform SVD on.
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.V = None

    def fit(self,matrix):
        """
        Calculate the SVD of the matrix using NumPy.

        Args:
        - matrix (numpy.ndarray): The matrix to perform SVD on
        """
        self.U, self.S, self.V= np.linalg.svd(matrix)

        
    def fit_transform(self,matrix):
        """
        Calculate the SVD of the matrix using NumPy.

        Args:
        - matrix (numpy.ndarray): The matrix to perform SVD on.
        
        Returns:
        - Array numpy.ndarrays: transformed matrix with componentes defined or 2 default
        """
        
        self.U, self.S, self.V = np.linalg.svd(matrix)
        matrix_transform = self._reconstruct(self.U, self.S, self.V, self.n_components)
        return matrix_transform
    
    def _reconstruct(self, U, S, V, k):
        """
        Reconstruct the original matrix from the SVD components using the first k components.
        
        Args:
        - U (numpy.ndarray): The U matrix from SVD.
        - S (numpy.ndarray): The S matrix from SVD.
        - V (numpy.ndarray): The V matrix from SVD.
        - k (int): The number of components to use for reconstruction.
        
        Returns:
        - numpy.ndarray: The reconstructed matrix.
        """
        S_k = np.zeros((k, k))
        S_k[:k, :k] = np.diag(S[:k])
        return np.dot(U[:, :k], np.dot(S_k, V[:k, :]))
