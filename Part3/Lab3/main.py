import numpy as np


class Question1(object):
    def pca_reduce_dimen(self, X):
        K = 2
        n = X.shape[1]
        
        X_mean = X.mean(axis=1).reshape((X.shape[0], 1))
        C = (X-X_mean) @ (X-X_mean).T / n
        eigen_value, eigen_vector = np.linalg.eigh(C)

        W = np.fliplr(eigen_vector).T
        s = np.flip(eigen_value)

        out = W[:K] @ X

        return out

    def pca_project(self, X, k):
        n = X.shape[1]
        L = X.shape[0]

        X_mean = X.mean(axis=1).reshape((X.shape[0], 1))
        C = (X-X_mean) @ (X-X_mean).T / n
        eigen_value, eigen_vector = np.linalg.eigh(C)

        W = np.fliplr(eigen_vector).T
        s = np.flip(eigen_value)

        topK_W = W[:k]

        filtered = topK_W.T @ (topK_W @ X)

        return filtered


class Question2(object):
    def wiener_filter(self, data_noisy, C, mu, sigma):
        filtered = None
        print()
        return filtered


class Question3(object):
    def embedding(self, A):
        n = A.shape[0]
        eig_val, eig_vec = np.linalg.eigh(A)
        sorted_idx = (-eig_val).argsort()
        eig_val = eig_val[sorted_idx]
        eig_vec = eig_vec[sorted_idx]

        return eig_vec[:, :n-1], eig_val
