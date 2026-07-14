import sys
import os

sys.path.append(os.path.abspath(".."))
from config import DATA_VARIANCE_SCALE, DATAPOINT_SIZE

import numpy as np
import torch
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

PLOTTING = False
device = torch.device('cpu')
z_min = 0.01
z_max = 2.0
z_obs = np.geomspace(z_min, z_max, DATAPOINT_SIZE)  # Logarithmic spacing to cover wide range of distances
F0 = 1e14  # Reference flux corresponding to zero distance modulus (arbitrary scale)

def create_synthetic_supernovae_dataset(n_samples=1000, h0s=None, print_status=True, data_var=DATA_VARIANCE_SCALE):
    '''
    Create n_samples numpy datapoints with Hubble Constant H0 = h0s for the respective samples and
    data_var variance of flux noise

    Inputs:
    - n_samples(required): No. of samples to be created
    - h0s(optional): Used to specify the Hubble Constant for the respective samples
        -> Must be a list, even for a single sample creation
    - print_status(optional): True prints "Creating sample x of XX"
    - data_var(optional): The variance of the AWGN noise in flux-space

    Outputs:
    - z_obs - the geomspace() list of the redshift bins
    - dataset - the actual noisy dataset
    - mu_noiseless - the noiseless dataset
    - h0_dataset - h0s for the respective samples (useful in case h0s not specified)
    '''
    # 1. Generate Redshifts (z) uniformly in log-space

    # 2. Calculate "True" Distance Modulus (theoretical signal)
    flux_data = []
    mu_noiseless = []
    h0_data = []
    cosmo = FlatLambdaCDM(H0=1. * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    mu_true = cosmo.distmod(z_obs).value
    flux_true_tild = F0*10**(-0.4*mu_true)  # Convert distance modulus to flux
    
    for i in range(n_samples):
        if print_status:
            print(f'\rCreating sample: {i+1} of {n_samples}', end='')
        if h0s is not None:
            H0 = h0s[i]
        else:
            H0 = 70 + np.random.normal(0, 5)
        # Add some variation to H0 for each sample

        flux_true = flux_true_tild * (H0**2)

        sigma_noise = data_var # * np.median(flux_true)

        for _ in range(50):  # Try multiple times to ensure positive flux
            flux_noise = np.random.normal(0, sigma_noise, DATAPOINT_SIZE)  # Additive Gaussian noise in flux space
            flux_obs = flux_true + flux_noise  # Noisy observed flux
            if np.all(flux_obs > 0):
                break
            flux_obs = np.clip(flux_obs, 1e-10, None)  # Ensure no negative flux values

        mu_obs = -2.5 * np.log10(flux_obs/F0)  # Convert back to distance modulus

        if PLOTTING:
            plt.scatter(z_obs, flux_true, alpha=0.5, label="True")  # Plot the true distance modulus for each sample (optional)
            plt.scatter(z_obs, flux_noise, alpha=0.5, label="Noise")  # Plot the true distance modulus for each sample (optional)
            plt.scatter(z_obs, flux_obs, alpha=0.5, label="Observed")  # Plot the true distance modulus for each sample (optional)
            plt.xlabel("Redshift (z)")
            plt.ylabel("Flux")
            plt.title(f"Sample {i+1} | H0: {H0:.2f}")
            plt.legend()
            plt.show()

            plt.scatter(z_obs, mu_true, alpha=0.5, label="True")  # Plot the true distance modulus for each sample (optional)
            plt.scatter(z_obs, mu_obs, alpha=0.5, label="Observed")  # Plot the true distance modulus for each sample (optional)
            plt.xlabel("Redshift (z)")
            plt.ylabel("Distance Modulus (mu)")
            plt.title(f"Sample {i+1} | H0: {H0:.2f}")
            plt.legend()
            plt.show()

        flux_data.append(flux_obs)
        mu_noiseless.append(mu_true)
        h0_data.append(H0)

    dataset = np.array(flux_data)
    mu_noiseless = np.array(mu_noiseless)
    h0_dataset = np.array(h0_data)
    if print_status:
        print('')
    return z_obs, dataset, mu_noiseless, h0_dataset

def create_torch_supernovae_dataset(n_train=1000, n_test=100, h0s_train=None, h0s_test=None):
    z_obs, flux, mu_noiseless, h0 = create_synthetic_supernovae_dataset(n_train, h0s=h0s_train)
    _, flux_test, mu_noiseless_test, h0_test = create_synthetic_supernovae_dataset(n_test, h0s=h0s_test)

    sims_data = np.zeros((n_train, (DATAPOINT_SIZE+1)), dtype=object)
    sims_data[:, :DATAPOINT_SIZE] = flux
    sims_data[:, DATAPOINT_SIZE] = h0
    sims_data = torch.from_numpy(sims_data.astype(np.float32)).to(device)

    test_data = np.zeros((n_test, (DATAPOINT_SIZE+1)), dtype=object)
    test_data[:, :DATAPOINT_SIZE] = flux_test
    test_data[:, DATAPOINT_SIZE] = h0_test
    test_data = torch.from_numpy(test_data.astype(np.float32)).to(device)

    return sims_data, test_data, z_obs
