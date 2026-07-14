import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np

from SupernovaDataset.config import DATAPOINT_SIZE, DATA_VARIANCE_SCALE

def supernova_ba_bound(log_prob_func, compressed_data, h02_data):
    log_prob = log_prob_func(y = h02_data, z=compressed_data)
    return log_prob.mean()

