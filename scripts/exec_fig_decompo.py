###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################


#          Avant de lancer le script, s'assurer d'avoir executé:
#
#                  module load python3/miniconda3
#                    source activate base_plus
#                 module load python3/outils-divers
#               module load development/python3-rpn

##########################################################################
#                  Importation des modules et libraries                  #
##########################################################################

import parametres

#from decomposition import lecture_concat
#from decomposition import index
from decomposition import calculate_variables
#from decomposition import difference
#from decomposition import erreurs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib 

#########################################################################
#

#########################################################################

cmap_data = ['#236c6c', '#348d8b', '#4eb2ab', '#77c1ad', '#cdd1aa', '#fbd1a2', '#f39f66', '#e07d40', '#c66122', '#a64a11']
cmap_precip = mcolors.ListedColormap(cmap_data, 'precipitation')

#########################################################################
#
#########################################################################

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
    ax[0,0].set_ylabel('$\omega$ [Pa s$^{-1}$]')#, size = label_size)
    ax[0,0].set_xlabel('W [mm]')#, size = label_size)
    ax[0,0].set_yticks(yticks)#,fontsize = label_size)
    ax[0,0].set_xticks(xticks)#,fontsize = label_size)
    ax[0,0].set_title('a)', loc='left', pad=20)
    ax[0,0].set_ylim(-3.1, 1.3)
    cb1 = fig.colorbar(p1, ax=ax[0,0], label='[mm]')
    #cb1.set_label(label='$\widetilde{PT}$ [mm]')#, size=label_size)
    #cb1.ax.tick_params(labelsize=label_size)
    
    #cmap = matplotlib.cm.get_cmap(cmap_)
    p2 = ax[0,1].pcolormesh(bins_tcwv, bins_w, values[1], cmap = cmap, norm=matplotlib.colors.LogNorm(), vmin = 0.0001, vmax=5000)
    ax[0,1].set_ylabel('$\omega$ [Pa s$^{-1}$]')#, size = label_size)
    ax[0,1].set_xlabel('W [mm]')#, size = label_size)
    ax[0,1].set_yticks(yticks)#,fontsize = label_size)
    ax[0,1].set_xticks(xticks)#,fontsize = label_size)
    ax[0,1].set_title('b)', loc='left', pad=20)
    ax[0,1].set_ylim(-3.1, 1.3)
    cb2 = fig.colorbar(p2, ax=ax[0,1], label = '[h]')
    #cb2.set_label(label='N')#, size=label_size)
    #cb2.ax.tick_params(labelsize=label_size)
    
    p3 = ax[1,0].pcolormesh(bins_tcwv, bins_w, values[2], cmap = cmap, vmin = 0, vmax = 1)
    ax[1,0].set_ylabel('$\omega$ [Pa s$^{-1}$]')#, size = label_size)
    ax[1,0].set_xlabel('W [mm]')#, size = label_size)
    ax[1,0].set_yticks(yticks)#,fontsize = label_size)
    ax[1,0].set_xticks(xticks)#,fontsize = label_size)
    ax[1,0].set_title('c)', loc='left', pad=20)
    ax[1,0].set_ylim(-3.1, 1.3)
    cb3 = fig.colorbar(p3, ax=ax[1,0], label = '')
    #cb3.set_label(label='p')#, size= label_size)
    #cb3.ax.tick_params(labelsize=label_size)
    
    p4 = ax[1,1].pcolormesh(bins_tcwv, bins_w, values[3], cmap = cmap, norm=matplotlib.colors.LogNorm(), vmin = 0.05, vmax=10)
    ax[1,1].set_ylabel('$\omega$ [Pa s$^{-1}$]')#, size= label_size)
    ax[1,1].set_xlabel('W [mm]')#, size = label_size)
    ax[1,1].set_yticks(yticks)#,fontsize = label_size)
    ax[1,1].set_xticks(xticks)#,fontsize = label_size)
    ax[1,1].set_title('d)', loc='left', pad=20)
    ax[1,1].set_ylim(-3.1, 1.3)
    cb4 = fig.colorbar(p4, ax=ax[1,1], label ='[mm h$^{-1}$]')
    #cb4.set_label(label='$\overline{I}_{rain}$ [mm/hr]')#, size=label_size)
    #cb4.ax.tick_params(labelsize=label_size)
    #cb.set_label(label=name, size='large')
    #cb.ax.tick_params(labelsize='large')
    #plt.title(donnees+': '+debut+' à '+fin)
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

    #plt.savefig(parametres.params.chemin_decomposition+'figures/'+name+'_'+debut+'_'+fin)
    plt.savefig('/transfert/lahaie/'+name+'_'+debut+'_'+fin+'.png')

    plt.show()
    return None

#########################################################################
#  Initialisation pour tracer figures decompo et decompo-erreur         #
#########################################################################

debut='2016-01'
fin='2018-01'
#debut = 'DJF'
#fin = 'DJF'
#periode = np.array(['2016-12', '2016-01', '2016-02', '2017-12', '2017-01', '2017-02'], dtype='datetime64[M]').astype(datetime.datetime)

#simulations = ['ERA5_GEMv5.0_CLASS_12km_P3_DEEPon', 'ERA5_GEMv5.0_ISBA_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff', 'ERA5_GEMv5.0_CLASS_12km_SU_DEEPon', 'ERA5_GEMv5.0_ISBA_2.5km_DEEPon', 'GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff', 'REANALYSIS_ERA5', 'IMERG', 'NEXRAD_STAGE_IV']
#simulations = ['NEXRAD_STAGE_IV']
simulations = ['IMERG_V7']

s_dom ='USA'
for produit in simulations:
    periode  = np.arange(debut, fin, dtype='datetime64[M]').astype(datetime.datetime)

    data = np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
    
    for i in range(1, len(periode)):
        data += np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
    
    precipitation, num, intensity, probability = calculate_variables(data, s_dom)
    figure_decompo([precipitation, num, probability, intensity], produit+s_dom, debut, fin, produit)
    print(num)
    #print(np.nanmax(precipitation), np.nanmin(precipitation))
    #print(np.nanmax(num), np.nanmin(num))
    #print(np.nanmax(probability), np.nanmin(probability))
    #print(np.nanmax(intensity), np.nanmin(intensity))
