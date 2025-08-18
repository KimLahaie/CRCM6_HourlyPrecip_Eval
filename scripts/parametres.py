import numpy as np
import xarray as xr

class params:
    # PATHS
    SAVE_PATH = '/pampa/lahaie/Decompo/post_traitements_20250130/'
    DECOMPO_PATH = '/pampa/lahaie/Decompo/decomposition_20250808/'

   # DOMAIN AND SUB-DOMAINS
    SUBDOM_MASK = xr.open_dataset('/pampa/lahaie/Decompo/mask_conus_spinup.nc').mask_conus
    SUBDOM = {'USA': {'number': 1, 'size': 12960}}

    # THRESHOLDS
    SEUIL_PR = 1/24
    SEUIL_EVENTS = 5

    # BINS
    BINS_TCWV = np.arange(0, 90, 5)
    BINS_W = np.arange(-3.1, 2.3, 0.2)
    #BINS_W = np.arange(-6.2, 3.0, 0.4)
    # Modèles, Réanalyses, Observations
    produits = {'SIMon': {'provenance_tcwv_w': 'SIMon',  'comparaisons': ['IMERG', 'ERA5', 'SIMoff', 'NEXRAD']},
                'SIMoff':{'provenance_tcwv_w': 'SIMoff', 'comparaisons': ['IMERG', 'ERA5', 'SIMon', 'NEXRAD']},
                'IMERG': {'provenance_tcwv_w': 'ERA5',   'comparaisons': ['NEXRAD']},
                'ERA5':  {'provenance_tcwv_w': 'ERA5',   'comparaisons': ['IMERG', 'NEXRAD']},
                'NEXRAD':{'provenance_tcwv_w': 'ERA5',   'comparaisons': []}}

