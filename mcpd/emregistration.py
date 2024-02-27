import numpy as np
from tqdm import tqdm

EPS = np.finfo(float).eps


def initialize_sigma2(X, Y):
    (batch, N, D) = X.shape
    (M, _) = Y.shape
    diff = X[:, None, :, :] - Y[None, :, None, :]
    err = np.sum(diff ** 2)
    return err / (batch * M * N * D)
    

class EMRegistration(object):

    def __init__(self, X, Y, Z, adjW, max_iterations=1000, tolerance=0.001, vis_interval=100, vis=True, *args, **kwargs):

        self.X = X
        self.Y = Y
        
        self.T, self.N, self.D = X.shape
        self.M, _ = Y.shape
        _, self.K = Z.shape
        self.sigma2 = initialize_sigma2(X, Y)
        
        self.TY = np.tile(Y[None, None, :, :], (self.T, self.K, 1, 1))

        self.tolerance = tolerance
        self.max_iterations =  max_iterations
        self.vis_interval = vis_interval
        self.vis = vis

        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.epsilon = 1.0

        self.Z = Z
        self.adjW = adjW
        
    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        for self.iteration in tqdm(range(self.max_iterations)):
            self.iterate()
            if callable(callback) and (self.iteration + 1) % self.vis_interval == 0 and self.vis:
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.Y, 
                          'TY': self.TY, 'Z': self.Z}
                callback(**kwargs)
        print("end iteration: ", self.iteration, self.diff, self.tolerance)
        return self.TY, self.get_registration_parameters()

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
        print("iteration: ", self.iteration)
        
    def weight_decaying(self):
        orgadjW = self.adjW / self.epsilon
        self.epsilon = max(self.epsilon * 0.99, 0.1)
        self.adjW = orgadjW * self.epsilon

    def expectation(self):
        T, K, M, N, D = self.T, self.K, self.M, self.N, self.D
        TY_tile = np.tile(self.TY[:, :, :, None, :], (1, 1, 1, N, 1))
        X_tile = np.tile(self.X[:, None, None, :, :], (1, K, M, 1, 1))    

        self.P = np.sum((X_tile - TY_tile) ** 2, -1) # T x K x M x N
        self.P = np.transpose(self.P, (0, 2, 3, 1))  # T x M x N x K

        self.P = np.exp(-self.P / (2 * self.sigma2))
        self.P = self.P + EPS

        Z_tile = np.tile(self.Z[None, :, None, :], (self.T, 1, self.N, 1))
        self.P = np.multiply(self.P, Z_tile)

        den = np.sum(self.P, axis=(1, 3)) # T x N
        den = den + EPS
        den = np.tile(den[:, None, :, None], (1, M, 1, K))
        
        self.P = np.divide(self.P, den) # T x M x N x K
        self.Pt1 = np.sum(self.P, axis=1) # T x N x K
        self.P1 = np.sum(self.P, axis=2) # T x M x K 
        self.Npk = np.sum(self.P, axis=(0, 1, 2)) # K
        self.Np = np.sum(self.P) # 1
        
    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()