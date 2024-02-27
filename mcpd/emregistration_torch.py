import numpy as np
import torch
from tqdm import tqdm

EPS = np.finfo(float).eps


def initialize_sigma2(X, Y):
    (batch, N, D) = X.size()
    (M, _) = Y.size()
    diff = X.view(batch, 1, N, D).expand(batch, M, N, D) - Y.view(1, M, 1, D).expand(batch, M, N, D)
    err = torch.sum(diff ** 2)
    return err / (batch * M * N * D)
    
    
class EMRegistration(object):
    
    def __init__(self, X, Y, Z, adjW, max_iterations=1000, tolerance=0.001, vis_interval=100, vis=True, *args, **kwargs):

        self.X = torch.FloatTensor(X).cuda()
        self.Y = torch.FloatTensor(Y).cuda()
        
        self.T, self.N, self.D = X.shape
        self.M, _ = Y.shape
        _, self.K = Z.shape
        
        self.sigma2 = initialize_sigma2(self.X, self.Y)
        self.sigma2 = self.sigma2.cpu().numpy()
        
        self.TY = self.Y.reshape(1, 1, self.M, self.D).repeat(self.T, self.K, 1, 1) # T x K x M x D
        
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.vis_interval = vis_interval
        self.vis = vis

        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.epsilon = 1.0
        
        self.Z = torch.FloatTensor(Z).cuda()
        self.adjW = torch.FloatTensor(adjW).cuda()

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        for self.iteration in tqdm(range(self.max_iterations)):
            self.iterate()
            if callable(callback) and (self.iteration + 1) % self.vis_interval == 0 and self.vis:
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X.cpu().numpy(), 'Y': self.Y.cpu().numpy(), 
                          'TY': self.TY.cpu().numpy(), 'Z': self.Z.cpu().numpy()}
                callback(**kwargs)

        return self.TY.cpu().numpy(), self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        T, K, M, N, D = self.T, self.K, self.M, self.N, self.D
        TY_tile = self.TY.contiguous().view(T, K, M, 1, D).expand(T, K, M, N, D)
        X_tile = self.X.contiguous().view(T, 1, 1, N, D).expand(T, K, M, N, D)          
        
        self.P = torch.sum((X_tile - TY_tile) ** 2, -1) # T x K x M x N
        self.P = self.P.permute(0, 2, 3, 1) # T x M x N x K

        self.P = torch.exp(-self.P / (2 * self.sigma2))
        self.P = self.P + EPS
        
        Z_tile = self.Z.contiguous().view(1, M, 1, K).expand(T, M, N, K) # T x M x N x K
        self.P = torch.mul(self.P, Z_tile) # T x M x N x K
        
        den = torch.sum(self.P, (1, 3)) # T x N
        den = den + EPS
        den = den.reshape(T, 1, N, 1).repeat(1, M, 1, K)
        
        self.P = torch.div(self.P, den) # T x M x N x K
        self.Pt1 = torch.sum(self.P, axis=1) # T x N x K
        self.P1 = torch.sum(self.P, axis=2) # T x M x K 
        self.Npk = torch.sum(self.P, (0, 1, 2)) # K
        self.Np = torch.sum(self.P) # 1
        
    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()