import numpy as np


class Lab1(object):
    def solver(self, A, b):
        return np.dot(np.linalg.inv(A), b)

    def fitting(self, x, y):
        onesMat = np.ones(x.shape)
        tmp = np.column_stack((x, onesMat))

        coeff = np.dot(np.linalg.pinv(tmp), y)
        
        return coeff

    def naive5(self, X, A, Y):
        # Calculate the matrix with $(i,j$)-th entry as  $\mathbf{x}_i^\top A \mathbf{y}_j$ by looping over the rows of $X,Y$.
        result = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                result[i][j] = np.dot(np.dot(x, A), y)

        return result

    def matrix5(self, X, A, Y):
        # Repeat part (a), but using only matrix operations (no loops!).
        return np.dot(np.dot(X, A), Y.transpose())

    def naive6(self, X, A):
        # Calculate a vector with $i$-th component $\mathbf{x}_i^\top A \mathbf{x}_i$ by looping over the rows of $X$.
        result = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            result[i] = np.dot(np.dot(x, A), x)
        return result

    def matrix6(self, X, A):
        # Repeat part (a) using matrix operations (no loops!).
        return np.sum(np.dot(X, A) * X, axis=1)
