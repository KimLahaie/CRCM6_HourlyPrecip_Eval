import numpy as np
import xarray as xr

class params:
    # PATHS
    INITIAL_DATA_PATH = 'PATH_TO_INITIAL_DATA'
    DECOMPO_PATH = 'PATH_TO_SAVE_BINNED_DATA'

   # DOMAIN AND SUB-DOMAINS
    SUBDOM_MASK = xr.open_dataset('PATH_TO_MASK_OF_DOMAIN').mask_conus
    SUBDOM = {'USA': {'number': 1, 'size': 12960}}

    # THRESHOLDS
    SEUIL_PR = 1/24
    SEUIL_EVENTS = 5

    # BINS
    BINS_TCWV = np.arange(0, 90, 5)
    BINS_W = np.arange(-3.1, 2.3, 0.2)

