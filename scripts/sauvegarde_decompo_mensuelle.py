###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################

##########################################################################
#                  Importation des modules et libraries                  #
##########################################################################

import parametres

from decomposition import assign_bins
from decomposition import decompose_regime

import xarray as xr
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
import pandas as pd

#########################################################################
#  Initialisation pour tracer figures decompo et decompo-erreur         #
#########################################################################

# Path to save the data
save_path = parametres.params.DECOMPO_PATH

# Path to retrieve the data
data_path = parametres.params.SAVE_PATH

# Selection of bins
bins_w = parametres.params.BINS_W
bins_tcwv = parametres.params.BINS_TCWV

t0 = time.time()

t1 = time.time()

# Iterate through subdirectories
for subdir in os.listdir(data_path):
    sub_path = os.path.join(data_path, subdir) 
    #if subdir in ['NEXRAD_STAGE_IV']:
        # Search for files corresponding to "_w", "_pr" and "_tcwv" in the subdirectory
    w_files = glob.glob(os.path.join(sub_path, f'*{subdir}*w.nc4'))
    pr_files = glob.glob(os.path.join(sub_path, f'*{subdir}*pr.nc4'))
    tcwv_files = glob.glob(os.path.join(sub_path, f'*{subdir}*tcwv.nc4'))
        # Iterate through files for each date
        
    for pr_file in pr_files:
            
        date = pr_file.split('_')[-2]  # Extract the date from the file name
        product = pr_file.split('/')[-1].split('_')[:-2]
        product = '_'.join(product)
        w_file = next((w for w in w_files if date in w), None)
        tcwv_file = next((tcwv for tcwv in tcwv_files if date in tcwv), None)
            
        if w_file is None:
            tcwv_data = xr.open_dataset(data_path + 'REANALYSIS_ERA5/REANALYSIS_ERA5_'+date+'_tcwv.nc4')
            w_data = xr.open_dataset(data_path + 'REANALYSIS_ERA5/REANALYSIS_ERA5_'+date+'_w500.nc4')
        else: 
            w_data = xr.open_dataset(w_file)
            tcwv_data = xr.open_dataset(tcwv_file)
            
        for sub_domain in parametres.params.SUBDOM:
            # Read files with xarray
            pr_data = xr.open_dataset(pr_file)
            pr_data = pr_data.where(parametres.params.SUBDOM_MASK == parametres.params.SUBDOM[sub_domain]['number'])
            tcwv_data = tcwv_data.where(parametres.params.SUBDOM_MASK == parametres.params.SUBDOM[sub_domain]['number'])
            w_data = w_data.where(parametres.params.SUBDOM_MASK == parametres.params.SUBDOM[sub_domain]['number'])

            index_w = assign_bins(w_data, bins_w)
            index_tcwv = assign_bins(tcwv_data, bins_tcwv)
            decomposition = decompose_regime(pr_data, index_tcwv, index_w, bins_tcwv, bins_w)

            # Specify the directory path to check/create
            directory_path = save_path + product + '/'

            # Check if the directory exists
            if not os.path.exists(directory_path):
                # Create the directory
                os.makedirs(directory_path)

            np.save(save_path + product + '/' + product + '_' + date + '_' + sub_domain, decomposition)
                
t2 = time.time()  # Stop the timer
print(" {}s (total: {}s)".format(t2-t1, t2-t0))

