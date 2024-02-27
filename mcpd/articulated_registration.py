import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class ArtRegistration():
    def __init__(self, source, targets, K, max_iterations=1000, vis_interval=500, vis=True, tolerance=1e-5, gpu=True):
        
        xs = np.copy(targets)
        if xs.ndim == 2:
            xs = xs[np.newaxis, :, :]
        ys = np.copy(source)
        gmm = GaussianMixture(n_components=K, covariance_type='full')
        gmm.fit(ys)
        pi = np.asarray(gmm.weights_)
        mu = np.asarray(gmm.means_)
        sigma = np.asarray(gmm.covariances_)

        M = ys.shape[0]
        Z = np.ones((M, K)) / K
        for k in range(K):
            Z[:, k] = pi[k] * multivariate_normal.pdf(ys, mean=mu[k], cov=sigma[k])
        Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))
        Z = Z.astype(np.float32)

        nbrs = NearestNeighbors(n_neighbors=21, algorithm='ball_tree').fit(ys)
        adjW = np.zeros((ys.shape[0], ys.shape[0]))
        indices = nbrs.kneighbors(ys, return_distance=False)
        for j in range(ys.shape[0]):
            adjW[j, indices[j, 1:]] = 1
        alpha = 1.0
        W = alpha * adjW
        W = W.astype(np.float32)
        
        if gpu:
            from .affine_part_registration_torch import AffinePartRegistration
        else:
            from .affine_part_registration import AffinePartRegistration
            
        self.reg = AffinePartRegistration(K=K, **{'X': xs, 'Y': ys, 'Z': Z, 'adjW': W, 'max_iterations': max_iterations, 
                                                  'vis_interval': vis_interval, 'vis': vis, 'tolerance': tolerance})
       
    def register(self, callback):
        TY, params = self.reg.register(callback)
        return TY, params