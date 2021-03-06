import numpy as np


class Procrustes():
    def __init__(self, scaling=True, reflection='best'):
        """
        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.
        :param scaling: if False, the scaling component of the transformation is forced
            to 1
        :param reflection: if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.
        """
        self.scaling = scaling
        self.reflection = reflection

    def fit(self, X, Y):
        """
        Fits the parameters of the linear transformation (translation,
        reflection, orthogonal rotation and scaling) to the (Y, X) pairs so that
        sum_of_squared_error(X, transformed(Y)) is minimal.
        :param X: target (model) coordinates (X and Y must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X)
        :param Y: input coordinates corresponding to X (X and Y must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X)
        :return:
            Z
                the matrix of transformed Y-values
            d
                the residual sum of squared errors, normalized according to a
                measure of the scale of X, ((X - X.mean(0))**2).sum()
        """

        nx, mx = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < mx:
            Y0 = np.concatenate((Y0, np.zeros(nx, mx - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if self.reflection != 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if self.reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if self.scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2

            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < mx:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        self.rotation = T
        self.scale = b
        self.translation = c

        return Z, d

    def transform(self, X):
        return (X @ self.rotation) * self.scale + self.translation


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.array([
        [2, 2],
        [4, 5],
        [10, 5]
    ])

    b = np.array([
        [9, 6],
        [3, 10],
        [3, 22],
        [9, 18]
    ])

    procrustes_transform = Procrustes()
    Z, d = procrustes_transform.fit(a, b[:3])
    b_transformed = procrustes_transform.transform(b)

    plt.scatter(b_transformed[:, 0], b_transformed[:, 1], color='red')
    plt.scatter(x=a[:, 0], y=a[:, 1], color='blue')
    plt.show()
