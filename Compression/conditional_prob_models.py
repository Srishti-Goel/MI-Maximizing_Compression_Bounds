import torch
import torch.nn as nn
import numpy as np

class ConditionalKDE(nn.Module):
    """
    A 2D conditional Gaussian-mixture density model over (z, y) pairs,
    used to evaluate log p(y | z) up to the mixing over z implied by
    the joint components (i.e. this models the joint p(z, y) mixture
    and treats it as a KDE-style conditional estimator).
 
    Each mixture component k has:
      - a center mu_k in R^2 (over the stacked (z, y) vector)
      - a 2x2 precision matrix P_k = L_k L_k^T, guaranteed symmetric
        positive-definite by construction (Cholesky parameterization)
      - a mixture weight pi_k, normalized via log_softmax so that
        sum_k pi_k = 1
    """
 
    def __init__(self, init_std, num_kde_points=10):
        '''
        init_std: Any variance scale for the initialization works, but exposing
                   this to prevent hardcoding.
        '''
        super().__init__()
        self.num_kde_points = num_kde_points
        self.dim = 2  # (z, y)
 
        # Parameterize the (lower-triangular) Cholesky factor L of the
        # precision matrix P = L @ L.T for each component.
        # Diagonal entries are stored in log-space to guarantee positivity
        # (and hence P is guaranteed positive-definite).
        # Off-diagonal (L[1, 0]) is a free real parameter.
        #
        # If we want E[P] ~ (1 / init_std) * I initially (i.e. covariance
        # ~ init_std * I), we need diag(L)^2 = 1 / init_std, so
        # log_diag_L = -0.5 * log(init_std).
        log_diag_init = -0.5 * np.log(init_std)
 
        self.log_diag_L = nn.Parameter(
            torch.full((num_kde_points, self.dim), log_diag_init)
        )  # (K, 2) -> diagonal entries of L, in log-space
        self.off_diag_L = nn.Parameter(
            torch.zeros(num_kde_points)
        )  # (K,) -> L[1, 0] entry (L[0, 1] stays 0 by construction)
 
        self.centers = nn.Parameter(torch.rand(num_kde_points, 2))
        # (num_kde_points, 2) for (z, y) pairs
 
        self.raw_weights = nn.Parameter(torch.rand(num_kde_points))
        # (num_kde_points,) unnormalized mixture logits; normalized via
        # log_softmax at use-time so the mixture weights are a valid
        # probability distribution.
 
    def _build_L(self):
        """
        Build the (K, 2, 2) batch of lower-triangular Cholesky factors.
        """
        diag = torch.exp(torch.clamp(self.log_diag_L, min=-10.0, max=10.0))
        # (K, 2), strictly positive
 
        L = torch.zeros(
            self.num_kde_points, 2, 2,
            dtype=diag.dtype, device=diag.device
        )
        L[:, 0, 0] = diag[:, 0]
        L[:, 1, 1] = diag[:, 1]
        L[:, 1, 0] = self.off_diag_L
        return L  # (K, 2, 2), lower-triangular, positive diagonal
 
    def log_prob(self, y, z):
        '''
        Calculate log p(y, z) under the Gaussian mixture (used as an
        estimate of the conditional log_prob(y | z) via the KDE joint).
        '''
        L = self._build_L()  # (K, 2, 2)
 
        vec = torch.column_stack([z, y])[:, None, :] - self.centers[None, :, :]
        # (batch_size, K, 2)
 
        # Compute L^T @ vec for each (batch, k): shape (batch, K, 2)
        # L is lower-triangular so L^T is upper-triangular.
        Lt_vec = torch.einsum('kde,bkd->bke', L, vec)
        # quad = vec^T P vec = vec^T L L^T vec = || L^T vec ||^2
        quad = torch.sum(Lt_vec ** 2, dim=-1)  # (batch, K)
 
        # log|Sigma_k| = -log|P_k| = -2 * sum(log(diag(L_k)))
        log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-12)  # (K, 2)
        half_log_det_P = torch.sum(log_diag, dim=-1)  # (K,)  = 0.5 * log|P_k|
 
        const = -0.5 * self.dim * np.log(2 * np.pi)
 
        log_component = const + half_log_det_P[None, :] - 0.5 * quad
        # (batch, K)
 
        log_weights = torch.log_softmax(self.raw_weights, dim=0)  # (K,), sums to 1 in prob space
 
        return torch.logsumexp(log_component + log_weights[None, :], dim=1)
        # (batch,) -- NOTE: no spurious division by batch size
 
    def forward(self, y, z):
        return self.log_prob(y, z)
 

class AutoregressiveTransform(nn.Module):
    """
    Autoregressive transformation using masked networks.
    Each dimension is transformed as: y_i = x_i * exp(s_i(x_{<i}, z)) + t_i(x_{<i}, z)
    where s_i and t_i are scale and translation networks conditioned on previous dimensions.
    """
    def __init__(self, y_dim=1, z_dim=1, hidden_dim=64, translate=0.0):
        super(AutoregressiveTransform, self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        
        # Network to output scale (log_s) and translation (t) for each dimension
        # Processes [previous dimensions + context] to output parameters
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        self.scale_layer = nn.Linear(hidden_dim, y_dim)
        # self.scale_layer.weight.data *= 1e-2
        self.translate_layer = nn.Linear(hidden_dim, y_dim)
        self.translate_layer.bias.data = torch.ones_like(self.translate_layer.bias.data) * translate    
    def forward(self, y, z):
        """
        Forward transformation: z_out = s(z, y) * y + t(z, y)
        Returns transformed y and log determinant of Jacobian.
        
        Args:
            y: Input of shape (batch_size, y_dim)
            z: Context of shape (batch_size, z_dim)
        
        Returns:
            y_transformed: Transformed output, shape (batch_size, y_dim)
            log_det_jac: Log determinant of Jacobian, shape (batch_size,)
        """
        h = self.net(z)
        scale = torch.tanh(self.scale_layer(h))  # log scale (for stability)
        translate = self.translate_layer(h)
        
        # Transformation: y_out = exp(scale) * y + translate
        y_transformed = torch.exp(scale) * y + translate
        
        # Log determinant of Jacobian (diagonal matrix for autoregressive)
        log_det_jac = scale.sum(dim=-1)
        
        return y_transformed, log_det_jac
    
    def inverse(self, y, z):
        """
        Inverse transformation.
        
        Args:
            y: Transformed variable of shape (batch_size, y_dim)
            z: Context of shape (batch_size, z_dim)
        
        Returns:
            y_original: Inverse transformed, shape (batch_size, y_dim)
            log_det_jac_inv: Log determinant of inverse Jacobian, shape (batch_size,)
        """
        h = self.net(z)
        scale = torch.tanh(self.scale_layer(h))
        translate = self.translate_layer(h)
        
        # Inverse: y_original = (y - translate) / exp(scale)
        y_original = (y - translate) / torch.exp(scale)
        
        # print(f'{scale.mean()=}')
        # print(f'{translate.mean()=}, {(y-translate).mean()=}')
        
        # Log det of inverse Jacobian is negative of forward
        log_det_jac_inv = -scale.sum(dim=-1)
        
        return y_original, log_det_jac_inv


class ConditionalNormalizingFlow(nn.Module):
    """
    Normalizing flow consisting of multiple autoregressive MAF transformations.
    Estimates p(y|z) by transforming a simple base distribution (Gaussian) through
    a sequence of invertible transformations.
    """
    def __init__(self, y_dim=1, z_dim=1, hidden_dim=64, num_flows=4):
        super(ConditionalNormalizingFlow, self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_flows = num_flows
        
        # Stack of autoregressive transformations
        self.transforms = nn.ModuleList([
            AutoregressiveTransform(y_dim=y_dim, z_dim=z_dim, hidden_dim=hidden_dim, translate= 0.0) #4900.0 if i == 0 else
            for i in range(num_flows)
        ])
        
        # Base distribution parameters (for reference)
        self.base_mean = nn.Parameter(torch.zeros(y_dim))
        # Initialize with larger std to avoid extreme log_probs during training startup
        self.base_log_std = nn.Parameter(torch.ones(y_dim) * torch.log(torch.tensor(100.0)))
    
    def forward(self, y, z):
        return self.log_prob(y,z)
    
    def inverse(self, y, z):
        """
        Inverse pass through the normalizing flow (reverse order).
        
        Args:
            y: Transformed variable of shape (batch_size, y_dim)
            z: Context of shape (batch_size, z_dim)
        
        Returns:
            y_original: Original variable
            total_log_det_jac_inv: Accumulated log determinant of inverse Jacobians
        """
        total_log_det_jac_inv = torch.zeros(y.shape[0], device=y.device)
        y_current = y
        
        # Apply inverse transformations in reverse order
        for transform in reversed(self.transforms):
            y_current, log_det = transform.inverse(y_current, z)
            total_log_det_jac_inv += log_det
            # print(f'{y_current.mean()=}')
        
        return y_current, total_log_det_jac_inv
    
    def log_prob(self, y, z):
        """
        Compute log p(y|z) using change of variables formula.
        log p(y|z) = log p(z_0|z) + sum of log |det J_i|
        where z_0 is the base distribution sample and J_i are the Jacobians.
        
        Args:
            y: Target variable of shape (batch_size, y_dim)
            z: Context of shape (batch_size, z_dim)
        
        Returns:
            log_prob: Log probability of shape (batch_size,)
        """
        # Transform to base distribution
        z0, log_det_jac = self.inverse(y, z)

        # print(f'{z0.mean()=}, {log_det_jac.mean()=}')
        
        # Base distribution log probability (diagonal Gaussian)
        base_mean = self.base_mean
        # print(f'{base_mean=}')
        base_std = torch.exp(self.base_log_std)
        # print(f'{base_std=}')
        
        # Proper log prob for diagonal Gaussian: sum over dims of [- 0.5*(z-μ)^2/σ^2 - log(σ)]
        # Then add normalization: - 0.5*d*log(2π)
        log_prob_base = -0.5 * ((z0 - base_mean) / base_std) ** 2 - self.base_log_std
        log_prob_base = log_prob_base.sum(dim=-1) - 0.5 * z0.shape[-1] * torch.log(torch.tensor(2.0 * torch.pi))
        # print(f'{log_prob_base=}, {z0.shape}')

        # Apply change of variables
        log_prob = log_prob_base + log_det_jac
        
        return log_prob
