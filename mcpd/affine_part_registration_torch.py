from builtins import super
import numpy as np
from .emregistration_torch import EMRegistration
import torch

EPS = np.finfo(float).eps


class AffinePartRegistration(EMRegistration):

    def __init__(self, R=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        t = np.zeros((self.T, self.K, self.D))
        self.t = torch.FloatTensor(t).cuda()
        
        R = np.zeros((self.T, self.K, self.D, self.D))
        for t in range(self.T):
            for k in range(self.K):
                R[t, k, :, :] = np.eye(3)
        self.R = torch.FloatTensor(R).cuda()

    def update_transform(self):
        self.X_hat = torch.zeros(self.T, self.K, self.N, self.D)
        self.YPY = torch.zeros(self.T, self.K)
        self.A = torch.zeros(self.T, self.K, self.D, self.D)
        
        for t in range(self.T):
            muX = torch.tensordot(self.P[t], self.X[t], dims=([1], [0])) # (M x N x K) x (N x D) = M x K x D
            muY = torch.tensordot(self.P[t], self.Y, dims=([0], [0])) # (M x N x K) x (M x D) = N x K x D
            muX = torch.sum(muX, 0) # K x D
            muY = torch.sum(muY, 0) # K x D

            NpK = torch.sum(self.P[t], (0, 1)) # K
            NpK = NpK.unsqueeze(1) # K x 1
            NpK = NpK.repeat(1, self.D) # K x D
            muX = torch.div(muX, NpK) # K x D
            muY = torch.div(muY, NpK) # K x D

            X_hat = self.X[t].unsqueeze(0) # 1 x N x D
            X_hat = X_hat.repeat(self.K, 1, 1) # K x N x D
            X_hat = X_hat - muX.unsqueeze(1).repeat(1, self.N, 1) # K x N x D
            self.X_hat[t] = X_hat
            
            Y_hat = self.Y.unsqueeze(0) # 1 x M x D
            Y_hat = Y_hat.repeat(self.K, 1, 1) # K x M x D
            Y_hat = Y_hat - muY.unsqueeze(1).repeat(1, self.M, 1) # K x M x D

            YPY = torch.sum(torch.mul(Y_hat, Y_hat), 2) # K x M
            YPY = torch.mul(self.P1[t].permute(1, 0), YPY) # K x M
            YPY = torch.sum(YPY, 1) # K
            self.YPY[t] = YPY
        
            for k in range(self.K):
                A = torch.tensordot(X_hat[k], self.P[t, :, :, k], dims=([0], [1])) # (N x D) x (M x N) = D x M
                A = torch.tensordot(A, Y_hat[k], dims=([1], [0])) # (D x M) x (M x D) = (D x D)
                self.A[t][k] = A
                U, _, V = np.linalg.svd(A.cpu().numpy(), full_matrices=True)
                C = np.ones((self.D, ))
                C[self.D-1] = np.linalg.det(np.dot(U, V))

                R = np.dot(np.dot(U, np.diag(C)), V)
                R = torch.FloatTensor(R).cuda()

                self.R[t][k] = R.permute(1, 0)
                self.t[t][k] = muX[k] - torch.tensordot(R, muY[k], dims=([1], [0]))
                
        Z = torch.sum(self.P, axis=(0, 2)) # M x K
        den = torch.sum(Z, 1, keepdims=True) # M x K
        adjW = torch.max(Z) * self.adjW / 2.0
        regul_Z = torch.tensordot(adjW, self.Z, dims=([1], [0])) # M x K
        regul_den = torch.sum(regul_Z, 1, keepdims=True) # M x K
        self.Z = torch.div(Z + regul_Z, den + regul_den)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            T, K, M, D = self.T, self.K, self.M, self.D
            ts = self.t.contiguous().view(T, K, 1, D).expand(T, K, M, D)
            Ys = self.Y.contiguous().view(1, 1, M, D).expand(T, K, M, D)
            self.TY = torch.matmul(Ys, self.R) + ts # T*K*M*D
            return
        else:
            return torch.dot(Y, self.R) + self.t

    def update_variance(self):
        qprev = self.q
        q = 0
        
        sigma2 = 0
        for t in range(self.T):
            for k in range(self.K):
                trAR = np.trace(np.dot(self.A[t][k].cpu().numpy(), self.R[t][k].cpu().numpy()))
                xPx = np.dot(np.transpose(self.Pt1[t, :, k].cpu().numpy()), np.sum(
                    np.multiply(self.X_hat[t][k].cpu().numpy(), self.X_hat[t][k].cpu().numpy()), axis=1))
                q += xPx - 2 * trAR + self.YPY[t][k].cpu().numpy()
                sigma2 += xPx - trAR

        qz = 0
        for t in range(self.T):
            qz += np.sum(-np.multiply(self.P1[t].cpu().numpy(), np.log(self.Z.cpu().numpy() + EPS)))

        self.q = q / (2 * self.sigma2) + qz + self.D * self.Np.cpu().numpy() / 2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = sigma2 / (self.Np.cpu().numpy() * self.D)
        if self.sigma2 <= self.tolerance:
            self.sigma2 = self.tolerance

    def get_registration_parameters(self):
        return self.R.cpu().numpy(), self.t.cpu().numpy(), self.Z.cpu().numpy()
