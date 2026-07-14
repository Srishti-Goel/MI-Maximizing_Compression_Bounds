import numpy as np

def moped(fid_vals, covariance, estimator_given_theta):
    '''
    Compute the MOPED compressed data vector and Fisher matrix.
    '''
    n_params = fid_vals.shape[0]
    n_data = covariance.shape[0]

    mu_init = estimator_given_theta(fid_vals)
    inv_cov = np.linalg.inv(covariance)
    b = np.zeros((n_params, n_data))
    F = np.zeros((n_params, n_params))

    # Compute numerical derivatives (central differences)
    delta = 1e-5
    derivatives = np.zeros((n_params, n_data))
    for i in range(n_params):
        theta_plus = fid_vals.copy()
        theta_minus = fid_vals.copy()
        theta_plus[i] += delta
        theta_minus[i] -= delta
        mu_plus = estimator_given_theta(theta_plus)
        mu_minus = estimator_given_theta(theta_minus)
        derivatives[i] = (mu_plus - mu_minus) / (2 * delta)
    
    # Compression vectors and Fisher matrix
    b[0] = inv_cov @ derivatives[0]
    norm = np.sqrt(derivatives[0] @ b[0])
    b[0] /= norm
    for i in range(1, n_params):
        proj = np.zeros(n_data)
        for j in range(i):
            proj += (derivatives[i] @ b[j]) * b[j]
        b[i] = inv_cov @ (derivatives[i] - proj)
        norm = np.sqrt((derivatives[i] - proj) @ b[i])

        b[i] /= norm
    
    print(b.shape)
    F = b @ inv_cov @ b.T
    return b, F, derivatives