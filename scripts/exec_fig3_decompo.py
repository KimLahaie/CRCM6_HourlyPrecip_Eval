###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################

##########################################################################
#                  Importation des modules et libraries                  #
##########################################################################

import parametres # Ensure it is correctly filled

from decomposition import calculate_variables

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib 

#########################################################################
#                     Plot configuration
#########################################################################

cmap_data = ['#236c6c', '#348d8b', '#4eb2ab', '#77c1ad', '#cdd1aa', '#fbd1a2', '#f39f66', '#e07d40', '#c66122', '#a64a11']
cmap_precip = mcolors.ListedColormap(cmap_data, 'precipitation')

font = {'family' : 'serif',
        'size'   : 24}
matplotlib.rc('font', **font)

def figure_decompo(values, name, debut, fin, titre, label_size = 24):
    fig, ax = plt.subplots(2, 2, figsize=[11,11], constrained_layout=True)    
    xticks = [0,25,50,75]
    yticks = [-3, -2, -1, 0, 1]
    bins_tcwv = parametres.params.BINS_TCWV
    bins_w = parametres.params.BINS_W
    
    cmap = matplotlib.cm.get_cmap(cmap_precip)
    cmap.set_bad(color='gray')

    p1 = ax[0,0].pcolormesh(bins_tcwv, bins_w, values[0], cmap = cmap, norm=matplotlib.colors.LogNorm(), vmin = 0.0001, vmax=500)
    ax[0,0].set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax[0,0].set_xlabel('W [mm]')
    ax[0,0].set_yticks(yticks)
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_title('a)', loc='left', pad=20)
    ax[0,0].set_ylim(-3.1, 1.3)
    cb1 = fig.colorbar(p1, ax=ax[0,0], label='[mm]')
    
    p2 = ax[0,1].pcolormesh(bins_tcwv, bins_w, values[1], cmap = cmap, norm=matplotlib.colors.LogNorm(), vmin = 0.0001, vmax=5000)
    ax[0,1].set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax[0,1].set_xlabel('W [mm]')
    ax[0,1].set_yticks(yticks)
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_title('b)', loc='left', pad=20)
    ax[0,1].set_ylim(-3.1, 1.3)
    cb2 = fig.colorbar(p2, ax=ax[0,1], label = '[h]')
    
    p3 = ax[1,0].pcolormesh(bins_tcwv, bins_w, values[2], cmap = cmap, vmin = 0, vmax = 1)
    ax[1,0].set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax[1,0].set_xlabel('W [mm]')
    ax[1,0].set_yticks(yticks)
    ax[1,0].set_xticks(xticks)
    ax[1,0].set_title('c)', loc='left', pad=20)
    ax[1,0].set_ylim(-3.1, 1.3)
    cb3 = fig.colorbar(p3, ax=ax[1,0], label = '')
    
    p4 = ax[1,1].pcolormesh(bins_tcwv, bins_w, values[3], cmap = cmap, norm=matplotlib.colors.LogNorm(), vmin = 0.05, vmax=10)
    ax[1,1].set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax[1,1].set_xlabel('W [mm]')
    ax[1,1].set_yticks(yticks)
    ax[1,1].set_xticks(xticks)
    ax[1,1].set_title('d)', loc='left', pad=20)
    ax[1,1].set_ylim(-3.1, 1.3)
    cb4 = fig.colorbar(p4, ax=ax[1,1], label ='[mm h$^{-1}$]')
    
    fig.suptitle(titre)
    ax[0,0].text(0.75,0.05 , '$\widetilde{PT}$' ,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[0,0].transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax[0,1].text(0.79,0.05 , '$N$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[0,1].transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax[1,0].text(0.8,0.05 , '$p$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[1,0].transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax[1,1].text(0.77,0.05 , '$\overline{I}^{rain}$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[1,1].transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    
    plt.savefig('/transfert/lahaie/fig3.png')
    plt.show()
    return None

#########################################################################
#                           Execution                                   #
#########################################################################

debut='2016-01'
fin='2018-01'
simulations = ['IMERG_V7']
s_dom ='USA'

for produit in simulations:
    periode  = np.arange(debut, fin, dtype='datetime64[M]').astype(datetime.datetime)

    data = np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
    
    for i in range(1, len(periode)):
        data += np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
    
    precipitation, num, intensity, probability = calculate_variables(data, s_dom)
    figure_decompo([precipitation, num, probability, intensity], produit+s_dom, debut, fin, produit)
