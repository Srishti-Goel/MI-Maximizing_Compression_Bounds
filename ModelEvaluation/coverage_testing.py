import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
import torch

from SupernovaDataset.dataset_creation import create_torch_supernovae_dataset
from SupernovaDataset.config import DATAPOINT_SIZE
from ModelEvaluation.abc import abc_testing, hist_with_error_bars

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def coverage_tests(num_samples, model, test_sims_data, method_name, select_percent=2.5e-2):
    coverage_sims_data, _, _ = create_torch_supernovae_dataset(num_samples, 1)
    coverage_sims_data = coverage_sims_data.to(device)
    # print(coverage_sims_data.device, device)
    model_obs = model(coverage_sims_data[:, :DATAPOINT_SIZE])
    model_test = model(test_sims_data[:, :DATAPOINT_SIZE])

    if type(model_obs) != np.ndarray:
        model_obs = model_obs.cpu().numpy()
        model_test = model_test.cpu().numpy()

    z_scores=[]
    pit_values=[]

    for i in range(num_samples):
        _, _, z_score, pit_value, _ = abc_testing(obs=model_obs[i, :], z_data=model_test,
                    y_data=test_sims_data[:, -1:].detach().cpu().numpy(),
                    true_param=coverage_sims_data[i, -1:].detach().cpu().numpy(),
                    param_name='$H_0$',
                    method_name=method_name,
                    plotting=False,
                    select_percent=select_percent
                    )
        # print(f'{deviation.item()=}')
        z_scores.append(z_score)
        pit_values.append(pit_value)

    z_scores = np.array(z_scores)
    pit_values = np.array(pit_values)

    hist_with_error_bars(z_scores, bins=30, xlabel='|True $\Theta − \mu | / \sigma$', ylabel='Density', title='Coverage / Calibration (z-score deviations)')
    x = np.linspace(0, max(z_scores), 200)
    plt.plot(x, halfnorm.pdf(x), 'r--', label='Ideal (half-normal)')
    plt.xlabel('|True $\Theta − \mu | / \sigma$')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Coverage / Calibration (z-score deviations)')
    plt.show()

    print(f"Fraction within 1σ: {np.mean(z_scores < 1):.3f}  (ideal ≈ 0.683)")
    print(f"Fraction within 2σ: {np.mean(z_scores < 2):.3f}  (ideal ≈ 0.954)")

    # 1. Histogram — should be flat/uniform if well-calibrated
    hist_with_error_bars(pit_values, bins=20, xlabel='PIT value', ylabel='Density', title=f'Coverage / Calibration Histogram ({method_name})')
    plt.axhline(1.0, color='red', linestyle='--', label='Ideal (uniform)')
    plt.xlabel('PIT value')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Coverage / Calibration Histogram ({method_name})')
    plt.show()

    # 2. Empirical CDF vs diagonal — classic SBC coverage plot
    sorted_pit = np.sort(pit_values)
    empirical_cdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)

    plt.plot(sorted_pit, empirical_cdf, label='Empirical')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal calibration')
    plt.xlabel('Credible level')
    plt.ylabel('Empirical coverage')
    plt.legend()
    plt.title('Coverage Test (SBC via PIT)')
    plt.show()
    return z_scores, pit_values