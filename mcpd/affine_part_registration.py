from builtins import super
import numpy as np
from .emregistration import EMRegistration

EPS = np.finfo(float).eps


def transpose_matrix(w=[0.0, 0.0, 1.0], theta=0.0):
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * np.dot(W, W)
    return R


class AffinePartRegistration(EMRegistration):

    def __init__(self, R=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        self.R = np.zeros((self.T, self.K, self.D, self.D))
        self.t = np.zeros((self.T, self.K, self.D))
        for t in range(self.T):
            for k in range(self.K):
                self.R[t, k, :, :] = np.eye(3)

    def update_transform(self):
        T, K, M, N, D = self.T, self.K, self.M, self.N, self.D
        self.X_hat = np.zeros((T, K, N, D), dtype=np.float32)
        self.YPY = np.zeros((T, K), dtype=np.float32)
        self.A = np.zeros((T, K, D, D), dtype=np.float32)

        for t in range(T):
            muX = np.tensordot(self.P[t], self.X[t], axes=([1], [0])) # (M x N x K) x (N x D) = M x K x D
            muY = np.tensordot(self.P[t], self.Y, axes=([0], [0])) # (M x N x K) x (M x D) = N x K x D
            muX = np.sum(muX, 0) # K x D
            muY = np.sum(muY, 0) # K x D

            NpK = np.sum(self.P[t], (0, 1)) # K
            NpK = NpK[:, None] # K x 1
            NpK = np.tile(NpK, (1, D)) # K x D
            muX = np.divide(muX, NpK) # K x D
            muY = np.divide(muY, NpK) # K x D

            X_hat = self.X[t][None, :, :] # 1 x N x D
            X_hat = np.tile(X_hat, (K, 1, 1)) # K x N x D
            X_hat = X_hat - np.tile(muX[:, None, :], (1, N, 1)) # K x N x D
            self.X_hat[t] = X_hat
            
            Y_hat = self.Y[None, :, :] # 1 x M x D
            Y_hat = np.tile(Y_hat, (K, 1, 1)) # K x M x D
            Y_hat = Y_hat - np.tile(muY[:, None, :], (1, M, 1)) # K x M x D

            YPY = np.sum(np.multiply(Y_hat, Y_hat), 2) # K x M
            YPY = np.multiply(self.P1[t].transpose(1, 0), YPY) # K x M
            YPY = np.sum(YPY, 1) # K
            self.YPY[t] = YPY
        
            for k in range(self.K):
                A = np.tensordot(X_hat[k], self.P[t, :, :, k], axes=([0], [1])) # (N x D) x (M x N) = D x M
                A = np.tensordot(A, Y_hat[k], axes=([1], [0])) # (D x M) x (M x D) = (D x D)
                self.A[t][k] = A
                U, _, V = np.linalg.svd(A, full_matrices=True)
                C = np.ones((self.D, ))
                C[self.D-1] = np.linalg.det(np.dot(U, V))

                R = np.dot(np.dot(U, np.diag(C)), V)

                self.R[t][k] = R.transpose(1, 0)
                self.t[t][k] = muX[k] - np.tensordot(R, muY[k], axes=([1], [0]))
                
        Z = np.sum(self.P, axis=(0, 2)) # M x K
        den = np.sum(Z, 1, keepdims=True) # M x K
        adjW = np.max(Z) * self.adjW / 2.0
        regul_Z = np.tensordot(adjW, self.Z, axes=([1], [0])) # M x K
        regul_den = np.sum(regul_Z, 1, keepdims=True) # M x K
        self.Z = np.divide(Z + regul_Z, den + regul_den)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            ts = np.tile(self.t[:, :, None, :], (1, 1, self.M, 1))
            Ys = np.tile(self.Y[None, None, :, :], (self.T, self.K, 1, 1))
            self.TY = np.matmul(Ys, self.R) + ts
            return
        else:
            return np.dot(Y, self.R) + self.t

    def update_variance(self):
        qprev = self.q
        q = 0
        
        sigma2 = 0
        for t in range(self.T):
            for k in range(self.K):
                trAR = np.trace(np.dot(self.A[t][k], self.R[t][k]))
                xPx = np.dot(np.transpose(self.Pt1[t, :, k]), np.sum(np.multiply(self.X_hat[t][k], self.X_hat[t][k]), axis=1))
                q += xPx - 2 * trAR + self.YPY[t][k]
                sigma2 += xPx - trAR

        qz = 0
        for t in range(self.T):
            qz += np.sum(-np.multiply(self.P1[t], np.log(self.Z + EPS)))

        self.q = q / (2 * self.sigma2) + qz + self.D * self.Np / 2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = sigma2 / (self.Np * self.D)
        if self.sigma2 <= self.tolerance:
            self.sigma2 = self.tolerance

    def get_registration_parameters(self):
        return self.R, self.t, self.Z