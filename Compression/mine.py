import numpy as np

import torch
import torch.nn as nn

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return torch.logsumexp(input, 0).squeeze() - np.log(input.shape[0])
    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + 1e-6) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.logsumexp(x, 0) - np.log(x.shape[0])
    t_exp = torch.exp(t_exp.clamp(max=20)).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T=None, y_dim=1, z_dim=1, hidden_dim=40, 
                 loss='mine', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha

        self.T = nn.Sequential(
            nn.Linear(y_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # no Tanh
            # nn.Tanh()
        )

    def forward(self, y, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(y.shape[0])]

        t = 1e1 * self.T(torch.concat((y, z), 1)).mean()
        t_marg = 1e1 * self.T(torch.concat((y, z_marg), 1))

        second_term, self.running_mean = ema_loss(
            t_marg, self.running_mean, self.alpha)

        return t - second_term

    def mi(self, y, z, z_marg=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = self.forward(y, z, z_marg)
        return mi
