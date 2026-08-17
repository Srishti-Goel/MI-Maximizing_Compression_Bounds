import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_binned_means_and_stds(z, ys,
                               num_bins=11,
                               plot_results=True,
                               print_results=True,
                               xlabel="Predicted Latent Variable",
                               title="Scatter Plot of Model Predictions vs True Parameter with Binned Means and Stds"
                            ):
    z_range = z.max() - z.min()
    window = 0.01 * z_range  # 1% of the entire range
    
    bin_edges = np.linspace(z.min(), z.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = []
    bin_stds = []
    bin_stds_err = []  # Error bars on the standard deviations
    
    for center in bin_centers:
        mask = (z >= center - window / 2) & (z <= center + window / 2)
        bin_ys = ys[mask]
        if len(bin_ys) > 0:
            bin_means.append(bin_ys.mean().item())
            std_val = bin_ys.std().item()
            bin_stds.append(std_val)
            
            # Calculate error on std deviation using SE(s) = s / sqrt(2(n-1))
            std_err = std_val / np.sqrt(2 * (len(bin_ys) - 1)) if len(bin_ys) > 1 else np.nan
            bin_stds_err.append(std_err)
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_stds_err.append(np.nan)
    
    # print(f"Expected std of true parameter Y within each bin (due to data noise): {expected_std:.2f}")
    if print_results:
        print(f"Dataset size: {len(ys)}, Number of bins: {num_bins}, Bin width: {window:.2f}")
        print(f"Bin centers: {bin_centers}")
        print(f"Bin means: {bin_means}")
        print(f"Bin stds: {bin_stds}")
        print(f"Bin stds errors: {bin_stds_err}")
    if plot_results:
        plt.scatter(z, ys, alpha=0.2, label="Model Predictions")
        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o", color="red", label="Binned mean ± std")
        plt.xlabel(xlabel)
        plt.ylabel("True parameter Y")
        plt.title(title)
        plt.legend()
        plt.show()

    return bin_centers, bin_means, np.array(bin_stds), np.array(bin_stds_err)

def evaluate_posterior_covariance(model, x_test, y_test, print_results=True, plot_results=True, num_bins=11):
    # model.eval()
    with torch.no_grad():
        z = model(x_test).detach().cpu().numpy()
        _, _, sample_bin_stds, sample_bin_stds_err = plot_binned_means_and_stds(
            z,
            y_test.cpu().numpy(),
            num_bins=num_bins,
            plot_results=plot_results,
            print_results=print_results,
        )

        mask = ~np.isnan(sample_bin_stds) & ~np.isnan(sample_bin_stds_err) & (sample_bin_stds > 0)

        if mask.sum() == 0:
            print("ERROR: No usable sample bins")
            print(f"{sample_bin_stds=} {sample_bin_stds_err}")

        sample_bin_stds = sample_bin_stds[mask]  # Exclude the first and last two bins
        sample_bin_stds_err = sample_bin_stds_err[mask]  # Exclude the first and last two bins

        # Inverse-variance weighted mean
        weights = 1.0 / np.power(sample_bin_stds_err, 2)  # Weight by inverse of squared errors

        # Combined estimate
        sigma_beta = np.sum(sample_bin_stds * weights) / np.sum(weights)

        # Uncertainty on the combined estimate
        sigma_beta_err = 1.0 / np.sqrt(np.sum(weights))

        return sigma_beta, sigma_beta_err, sample_bin_stds, sample_bin_stds_err
