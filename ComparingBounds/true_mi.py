import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np

from SupernovaDataset.config import DATA_VARIANCE_SCALE

def supernova_mi(
        simulate,
        prior_var=1,
        cond_var=DATA_VARIANCE_SCALE
    ):
    F_tild_i = simulate(h0=1)
    print(F_tild_i)
    
    n_dim = F_tild_i.shape[0]

    sigma_f = (prior_var * F_tild_i @ F_tild_i.T) + (cond_var * np.eye(n_dim))
    det_sigma_f = np.linalg.det(sigma_f)
    
    MI = (0.5 * np.log(det_sigma_f)) - (0.5 * n_dim * np.log(cond_var))
    return MI