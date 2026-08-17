import torch
import torch.nn as nn


class TUBA(nn.Module):
    def __init__(self, y_dim=1, z_dim=1, hidden_dim=400, a=None):
        super().__init__()

        # Critic f(theta, z): unbounded output (no Tanh!)
        self.T = nn.Sequential(
            nn.Linear(y_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Baseline a(z): must be strictly positive -> Softplus, not Tanh
        if a is None:
            self.a = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),
            )
        else:
            self.a = a

    def forward(self, y, z):
        """
        y: (batch, y_dim) samples of theta ~ p(theta)
        z: (batch, z_dim) samples of z, paired with y as p(theta, z)
        Assumes rows of y and z are already jointly sampled (y[i], z[i]) ~ p(theta,z),
        and that shuffling z against y gives (approximate) samples from p(theta)p(z).
        """
        batch_size = y.shape[0]

        # ---- joint term: E_{p(theta,z)}[f(theta,z)] ----
        f_joint = self.T(torch.cat([y, z], dim=1))          # (batch, 1)
        joint_term = f_joint.mean()

        # ---- all pairs f(y_i, z_j), i,j = 1..batch ----
        y_tile = y.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, y.shape[1])
        z_tile = z.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, z.shape[1])
        f_all = self.T(torch.cat([y_tile, z_tile], dim=1))  # (batch*batch, 1)
        f_all = f_all.view(batch_size, batch_size)           # f_all[i, j] = f(y_i, z_j)

        # E_{p(theta)}[exp f(theta, z_j)]  -> average over i (theta index), for each z_j
        # subtract max for numerical stability (does not change the mean-log-etc. below
        # since we're not taking a log of this particular term, just being safe with exp)
        log_mean_exp_f = torch.logsumexp(f_all, dim=0) - torch.log(
            torch.tensor(batch_size, dtype=f_all.dtype, device=f_all.device)
        )  # (batch,) = log E_{p(theta)}[exp f(theta, z_j)]

        a_z = self.a(z).squeeze(-1)          # (batch,)  a(z_j) > 0
        log_a_z = torch.log(a_z + 1e-12)

        # exp(f) / a(z) computed stably as exp(log_mean_exp_f - log_a_z)
        correction = torch.exp(log_mean_exp_f - log_a_z) + log_a_z - 1
        correction_term = correction.mean()  # E_{p(z)}[ ... ]

        tuba = joint_term - correction_term
        return tuba

class DV_with_Jensen(nn.Module):
    def __init__(self, y_dim=1, z_dim=1, hidden_dim=400):
        super().__init__()

        # Critic f(theta, z): unbounded output (no Tanh!)
        self.T = nn.Sequential(
            nn.Linear(y_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y, z, z_marg=None):
        """
        y: (batch, y_dim) samples of theta ~ p(theta)
        z: (batch, z_dim) samples of z, paired with y as p(theta, z)
        Assumes rows of y and z are already jointly sampled (y[i], z[i]) ~ p(theta,z),
        and that shuffling z against y gives (approximate) samples from p(theta)p(z).
        """
        if z_marg is None:
            z_marg = z[torch.randperm(y.shape[0])]

        
        t = 1e2 * self.T(torch.concat((y, z), 1)).mean()
        t_marg = 1e2 * self.T(torch.concat((y, z_marg), 1)).mean()

        return t - t_marg