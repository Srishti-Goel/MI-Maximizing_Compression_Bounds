import sys
import os

sys.path.append(os.path.abspath(".."))

import torch

def sampler(mean, logvar):
    '''
    Simple utility function for all compressors to sample given the output mean and logvariance from a gaussian distribution
    '''
    eps = torch.randn_like(logvar)
    return mean + eps * torch.exp(0.5*logvar)


class IBCompressor(torch.nn.Module):
    def __init__(self, encoder, x_dim, z_dim):
        '''
        Encoder: must be of the form torch.nn.Sequential()
        '''
        super().__init__()
        
        self._latent_dim = z_dim

        self.encoder = encoder

    def forward(self, x, be_verbose = False):
        '''
        Output: mean, logvar, sampled_z
        '''
        encoded = self.encoder(x)
        mean = encoded[:, :self._latent_dim]
        logvar = encoded[:, self._latent_dim:]
        z = sampler(mean, logvar)

        if be_verbose:
            print(f'Shapes while encoding: Mean: {mean.shape}, logvar: {logvar.shape}')
        
        return mean, logvar, z