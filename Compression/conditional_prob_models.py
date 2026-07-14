import torch
import torch.nn as nn

class ConditionalKDE(nn.Module):
    def __init__(self, init_std, num_kde_points=10):
        '''
        init_std: Any variance scale for the initialization works, but exposing this to prevent hardcoding
        '''
        super().__init__()

        log_h_init = torch.log(torch.ones(num_kde_points, 2, 2) * init_std)  # (num_kde_points, 2, 2)
        for i in range(num_kde_points):
            log_h_init[i, 0, 1] = log_h_init[i, 1, 0] = torch.log(torch.tensor(1.0))  # Set off-diagonal to zero (log(1) = 0)
        
        # Initialize with the precomputed values
        self.log_h = nn.Parameter(log_h_init)
        self.centers = nn.Parameter(torch.rand(num_kde_points, 2))
        # (num_kde_points, 2) for (z, y) pairs
        self.weights = nn.Parameter(torch.rand(num_kde_points))
        # (num_kde_points,) for the weights of each KDE point

    def log_prob(self, y, z):
        '''
        Calculate the log_prob(y | z)
        '''
        h = torch.exp(torch.clamp(self.log_h, max=10.0, min=1e-4))
        # (2, 2) covariance matrix for the joint KDE
        vec = torch.column_stack([z, y])[:, None, :] - self.centers[None, :, :]
        # (batch_size, num_kde_points, 2)

        log_k = -0.5 * torch.einsum(
            'bkd,kde,bke->bk',
            vec, h, vec
        )
        return torch.logsumexp(log_k + self.weights, dim=1)/(vec.shape[0]**2)
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
        self.scale_layer.weight.data *= 1e-2
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
        log_det_jac = scale.mean(dim=-1)
        
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
        log_det_jac_inv = -scale.mean(dim=-1)
        
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
            AutoregressiveTransform(y_dim=y_dim, z_dim=z_dim, hidden_dim=hidden_dim, translate=70.0 if i == 0 else 0.0)
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
        log_prob_base = log_prob_base.mean(dim=-1) # - 0.5 * z0.shape[-1] * torch.log(torch.tensor(2.0 * torch.pi))
        # print(f'{log_prob_base=}, {z0.shape}')

        # Apply change of variables
        log_prob = log_prob_base + log_det_jac
        
        return log_prob
