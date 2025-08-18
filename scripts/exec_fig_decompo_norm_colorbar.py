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

from decomposition import calculate_variables
from decomposition import calculate_errors
from decomposition import multiply_differences
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
#########################################################################
#
#########################################################################

font = {'family' : 'serif',
        'size'   : 24}
mpl.rc('font', **font)



# Step 1: Import packages
import numpy as np
import math
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
 
 
# Step 2: Define parameters of log-normal colorbar
 



def figure_decompo(values, name, debut, fin,titre):#norm=None):
    
    cmap_data = ["#45220d","#934616","#e07d40","#f0bea0","#FFFFFF","#80D9E5","#00b2ca","#16679a", "#113055"]

    yticks = [-6, -4, -2, 0, 2]
    xticks = [0, 25,50,75]

    cmap_precip = LinearSegmentedColormap.from_list('custom_cmap', cmap_data)
    #cmap_precip = mcolors.ListedColormap(cmap_data)
    bounds = [-1000,-100, -50, -25, -10,-0.000000000000000000000000000000000000004, 0.000000000000000000000000000000000000004, 10, 25, 50,100,1000]
    norm = BoundaryNorm(bounds, cmap_precip.N)
    print(norm)
#    color_bounds = [-40,-30,-20,-10,-0.01,0.1,10,20,30,40]
    #bounds = [ -25, -15, -10, -7.5, -5, -2.5, -0.5, 0.5, 2.5, 5,7.5, 10, 15,25]
    #bounds = [-25,-15,-10,-7.5,-5, -2.5, -1,1, 2.5,5, 7.5,10,15,25]
#    bounds_label = ['-10$^{2}$','-10$^{1}$', '-10$^{0}$', '-10$^{-1}$','0', '10$^{-1}$', '10$^{0}$', '10$^{1}$','10$^{2}$']
#    colorbar_norm = mpl.colors.BoundaryNorm(color_bounds, cmap_precip.N)#, extend='both')
    #bounds = [-0.3,-0.2,-0.1,-0.075,-0.05,-0.025,0.025,0.05,0.075,0.1,0.2, 0.3]
    #colorbar_norm = mpl.colors.BoundaryNorm(bounds, cmap_precip.N, extend='both')
    
    fig = plt.figure(figsize=(11, 11))#, constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], figure=fig)
    
    # Main plot setup (2x2 grid)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Custom Colorbar Setup
    cax = fig.add_subplot(gs[:, 2])  # Use entire right column for the colorbar

    bins_tcwv = parametres.params.BINS_TCWV
    bins_w = parametres.params.BINS_W

    cmap = mpl.cm.get_cmap(cmap_precip)
    cmap.set_bad(color='gray')

#    ticks = [ -10**(-1),-10**(0), -10**(1), -10**(2),0,10**2, 10**(1), 10**(0), 10**(-1)] 
#    ticks.sort()
#    nbr_of_ticks       = len(ticks)
#    orders_of_mag      = math.floor(nbr_of_ticks/2)       # orders of magnitude of the ticks; assumes that the positive and negative values are "symmetrical"
   #lin_threshold      = 10**(-1 * orders_of_mag)
#    lin_threshold      = 10**(-5)
#    print(lin_threshold)
#    lin_scale          = lin_threshold 
#    colorbar_norm = mpl.colors.SymLogNorm(vmin=ticks[0], vmax=ticks[-1], linthresh=lin_threshold, linscale=lin_scale, base=10)
   #colorbar_norm      = clrs.SymLogNorm(vmin=ticks[0], vmax=ticks[-1], linthresh=lin_threshold, linscale=lin_scale, clip=False, base=10)   # very important to set clip to False, or else out-of-bounds data will not appear on figure

    p1 = ax1.pcolormesh(bins_tcwv, bins_w, values[0], cmap=cmap, norm=norm)#, vmin = 0.0001, vmax=50)
    ax1.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax1.set_xlabel('W [mm]')
    ax1.set_yticks(yticks)
    ax1.set_xticks(xticks)
    ax1.set_title('a)', loc='left', pad=10)


    p2 = ax2.pcolormesh(bins_tcwv, bins_w, values[1], cmap = cmap, norm=norm)#, vmin = 0.0001, vmax= 685)
    ax2.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax2.set_xlabel('W [mm]')
    ax2.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_title('b)', loc='left', pad=10)

    p3 = ax3.pcolormesh(bins_tcwv, bins_w, values[2], cmap = cmap, norm=norm)#, vmin = 0, vmax = 1)
    ax3.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax3.set_xlabel('W [mm]')
    ax3.set_yticks(yticks)
    ax3.set_xticks(xticks)
    ax3.set_title('c)', loc='left', pad =10)
    

    p4 = ax4.pcolormesh(bins_tcwv, bins_w, values[3], cmap = cmap, norm=norm)#, vmin = 0.05, vmax= 23)
    ax4.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax4.set_xlabel('W [mm]')
    ax4.set_yticks(yticks)
    ax4.set_xticks(xticks)
    ax4.set_title('d)', loc='left', pad = 10)
    
    
    nom = name.split('USA', 1)
    fig.suptitle(titre)
    label_size=24
    ax1.text(0.75,0.05 , '$\widetilde{PT}^{\epsilon}_n$' ,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax1.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax2.text(0.79,0.05 , '$N^{\epsilon}_n$',
        horizontalalignment='left',
        verticalalignment='bottom', 
        transform=ax2.transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax3.text(0.8,0.05 , '$p^{\epsilon}_n$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax3.transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax4.text(0.83,0.05 , '$I^{\epsilon}_n$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax4.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)

    fig.subplots_adjust(left=0.11, right=0.85, top=0.86, bottom=0.09, wspace=0.60, hspace=0.35)

    # This is to ensure there are no leftover axes from GridSpec or previous calls
    for ax in fig.get_axes():
        if ax not in [ax1, ax2, ax3, ax4]:
            fig.delaxes(ax)  # Remove the unwanted axis
    ###Custum colorbar
# Set up a new axis for the custom colorbar
    cax = fig.add_axes([0.8, 0.1, 0.03, 0.8])  # Adjust position as needed

    heights = [0.25, 0.125, 0.075, 0.05, 0.005,0.05, 0.075, 0.125, 0.25]
    tick_positions = [0, 0.25, 0.375, 0.45, 0.5025, 0.555, 0.63, 0.755,1.005]
    tick_labels = ['-100', '-50', '-25', '10', '0', '10', '25', '50', '100']
    # Define custom heights for each interval (modify these values for custom spacing)
    start = 0

    # Draw rectangles for each color interval
    for color, height in zip(cmap_data, heights):
        rect = mpatches.Rectangle((0, start), 1, height, color=color, ec='none')  # ec='k' adds black edge for separation
        cax.add_patch(rect)
        start += height

    # Set axis limits to match the total height of all rectangles combined
    cax.set_xlim(0, 1)
    cax.set_ylim(0, start)  # Use 'start' to ensure the ylim matches the height of all rectangles

    # Add custom ticks and labels
    cax.set_yticks(tick_positions)
    cax.set_yticklabels(tick_labels)
    cax.yaxis.set_tick_params(width=1, labelright=True, labelleft=False, right=True, left=False)  # Optionally hide tick mar
    cax.xaxis.set_tick_params(width=0, labelright=False, labelleft=False)  # Optionally hide tick mar
    # Define custom heights for each interval (modify these values for custom spacing)
    cax.set_ylabel('[%]', rotation=-90, labelpad=-75, va='bottom')
    #plt.savefig(parametres.params.chemin_decomposition+'figures/'+name+'_'+debut+'_'+fin)
    plt.savefig('/transfert/lahaie/'+name+'_'+'.png')
    #plt.savefig('/transfert/lahaie/allo.png')
    #plt.close()
    plt.show()
    return None

def figure_residuel(residual, name, debut, fin):
    cmap_data = ["#45220d","#6a310f","#a64a11","#e07d40","#f9b27c","#ffffff","#73D4D4","#00b2ca","#0f80aa","#1d4e89", "#113055"]
    bounds = [-100,-50, -10, -5, -0.1, 0.1, 5, 10,50,100]
    bounds_label = [-100,-50, -10, -5, -0.1,0, 0.1, 5, 10,50,100]
    cmap_precip = mcolors.ListedColormap(cmap_data)
    norm = mpl.colors.BoundaryNorm(bounds, cmap_precip.N, extend='both')
    fig, ax = plt.subplots(1, 1, figsize=[3,3], constrained_layout=True)

    bins_tcwv = parametres.params.bins_tcwv
    bins_w = parametres.params.bins_w

    cmap = mpl.cm.get_cmap(cmap_precip)
    cmap.set_bad(color='gray')

    p1 = ax.pcolormesh(bins_tcwv, bins_w, residual, cmap = cmap, norm=norm)#, vmin = 0.0001, vmax=50)
    ax.set_ylabel('$\omega$ [Pa/s]')
    ax.set_xlabel('W [mm]')
    cb= fig.colorbar(p1, extend='both', ticks=bounds, spacing='proportional')
    cb.set_label(label='[mm]')
    cb.ax.tick_params(labelsize='large')
    fig.suptitle(name+'   '+debut+'_'+fin)
    ax.text(0.75,0.05 , '$R^{\epsilon}$' ,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    plt.savefig('/transfert/lahaie/'+'residuel'+name+'_'+debut+'_'+fin+".png")
    plt.close()
    #plt.show()
    return None

#########################################################################
#  Initialisation pour tracer figures decompo et decompo-erreur         #
#########################################################################

# SELECTION DE LA PERIODE
debut='2016-01'
fin='2018-01'
#debut = 'JJA'
#fin = 'JJA'
#periode = np.array(['2016-06', '2016-07', '2016-08', '2017-06', '2017-07', '2017-08'], dtype='datetime64[M]').astype(datetime.datetime)
#debut = 'DJF'#
#fin = 'DJF'
#periode = np.array(['2016-01', '2016-02', '2016-12', '2017-01', '2017-02', '2017-12'], dtype='datetime64[M]').astype(datetime.datetime)
#debut = 'MAM'
#fin = 'MAM'
#periode = np.array(['2016-03', '2016-04', '2016-05', '2017-03', '2017-04', '2017-05'], dtype='datetime64[M]').astype(datetime.datetime)
#debut = 'SON'
#fin = 'SON'
#periode = np.array(['2016-09', '2016-10', '2016-11', '2017-09', '2017-10', '2017-11'], dtype='datetime64[M]').astype(datetime.datetime)


#produits = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPon']
#comparaisons = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPon_SU']
#titres = ['GEM50-2.5-C-on-P3 [ERA5] -\n GEM50-2.5-C-on-SU [ERA5]']

# SELECTION DU PRODUIT
produits = ['GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff','ERA5_GEMv5.0_CLASS_2.5km_DEEPoff','ERA5_GEMv5.0_ISBA_2.5km_DEEPon']
comparaisons = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPoff',  'GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPon',  'ERA5_GEMv5.0_CLASS_2.5km_DEEPon']
titres = ['GEM50-2.5-C-off-P3 [GEM50-12-SU] -\n GEM50-2.5-C-off-P3 [ERA5]', 'GEM50-2.5-C-off-P3 [GEM50-12-P3] -\n GEM50-2.5-C-off-P3 [GEM50-12-SU]', 'GEM51-2.5-C-off-P3 [GEM51-12-P3] -\n GEM50-2.5-C-off-P3 [GEM50-12-P3]', 'GEM50-2.5-C-off-P3 [ERA5] -\n GEM50-2.5-C-on-P3 [ERA5]', 'GEM50-2.5-I-on-P3 [ERA5] -\n GEM50-2.5-C-on-P3 [ERA5]']
#produits = ['ERA5_GEMv5.0_CLASS_12km_SU_DEEPon','ERA5_GEMv5.0_CLASS_2.5km_DEEPon', 'NEXRAD_STAGE_IV', 'ERA5_GEMv5.0_CLASS_12km_P3_DEEPon', 'NEXRAD_STAGE_IV', 'ERA5_GEMv5.0_CLASS_12km_P3_DEEPon','ERA5_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff','ERA5_GEMv5.0_CLASS_2.5km_DEEPon_SU','ERA5_GEMv5.0_ISBA_2.5km_DEEPoff','ERA5_GEMv5.0_ISBA_2.5km_DEEPon', 'IMERG_V6']
#comparaisons = [ 'IMERG_V7', 'IMERG_V7', 'IMERG_V7', 'ERA5_GEMv5.0_CLASS_12km_SU_DEEPon', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPon', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPoff',  'GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff', 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPon', 'ERA5_GEMv5.0_ISBA_2.5km_DEEPon','ERA5_GEMv5.0_CLASS_2.5km_DEEPon', 'IMERG_V7']
#titres = ['GEM50-12-C-on-SU [ERA5] -\n IMERG V7', 'GEM50-2.5-C-on-P3 [ERA5] -\n IMERG V7', 'Stage IV - IMERG V7', 'GEM50-12-C-on-P3 [ERA5] -\n GEM50-12-C-on-SU [ERA5]', 'GEM50-2.5-C-off-P3 [ERA5] -\n GEM50-2.5-C-on-P3 [ERA5]', 'GEM50-2.5-C-off-P3 [GEM50-12-SU] -\n GEM50-2.5-C-off-P3 [ERA5]', 'GEM50-2.5-C-off-P3 [GEM50-12-P3] -\n GEM50-2.5-C-off-P3 [GEM50-12-SU]', 'GEM51-2.5-C-off-P3 [GEM51-12-P3] -\n GEM50-2.5-C-off-P3 [GEM50-12-P3]', 'GEM50-2.5-C-on-SU [ERA5] -\n GEM50-2.5-C-on-P3 [ERA5]', 'GEM50-2.5-I-off-P3 [ERA5] -\n GEM50-2.5-I-on-P3 [ERA5]', 'GEM50-2.5-I-on-P3 [ERA5] -\n GEM50-2.5-C-on-P3 [ERA5]', 'IMERG V6 - IMERG V7']
#

#produits = ['ERA5_GEMv5.0_CLASS_12km_SU_DEEPon', 'ERA5_GEMv5.0_CLASS_2.5km_DEEPon']  
#titres = ['GEM50-12-C-on-SU [ERA5] - IMERG (V7)', 'GEM50-2.5-C-on-P3 - IMERG (V7)']
#produits = ['ERA5_GEMv5.0_ISBA_2.5km_DEEPoff']           
#produits = ['GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff']
#produits = ['ERA5_GEMv5.0_CLASS_12km_SU_DEEPon']
#produits = ['ERA5_GEMv5.0_ISBA_2.5km_DEEPon']
#produits = ['GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff']
#produits = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPoff']
#produits = ['GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff']
#produits = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPon']#,'GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff']
#produits = ['NEXRAD_STAGE_IV']
#produits = ['IMERG_V6']
#produits = ['ERA5_GEMv5.0_CLASS_2.5km_DEEPon']

# SELECTION DU PODUIT DE COMPARAISON
#comparaison = 'IMERG_V7'
#comparaison2 = 'NEXRAD_STAGE_IV'
#comparaison3 = 'IMERG_V6'
#comparaison = 'ERA5_GEMv5.0_CLASS_12km_P3_DEEPon'  
#comparaison = 'ERA5_GEMv5.0_ISBA_2.5km_DEEPoff'           
#comparaison = 'GEM12_P3-CRI_GEMv5.1.1_CLASS_2.5km_DEEPoff'
#comparaison = 'ERA5_GEMv5.0_CLASS_12km_SU_DEEPon'
#comparaison = 'ERA5_GEMv5.0_ISBA_2.5km_DEEPon'
#comparaison = 'GEM12_SU_GEMv5.0_CLASS_2.5km_DEEPoff'
#comparaison = 'ERA5_GEMv5.0_CLASS_2.5km_DEEPoff'
#comparaison = 'GEM12_P3-CRI_GEMv5.0_CLASS_2.5km_DEEPoff'

s_dom = 'USA' 
for i in range(len(produits)):
        produit = produits[i]
        titre = titres[i]
        comparaison = comparaisons[i]
        periode  = np.arange(debut, fin, dtype='datetime64[M]').astype(datetime.datetime)

        data = np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        data2 = np.load(parametres.params.DECOMPO_PATH+comparaison+'/'+comparaison+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        for i in range(1, len(periode)):
            data += np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
            data2 += np.load(parametres.params.DECOMPO_PATH+comparaison+'/'+comparaison+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')

        precipitation, num, intensity, probability = calculate_variables(data, s_dom)
        precipitationo, numo, intensityo, probabilityo = calculate_variables(data2, s_dom)
         
        e_precip, e_num, e_probability, e_intensity, residual = calculate_errors(precipitation, num, intensity, probability, precipitationo,numo, intensityo, probabilityo)
        tot = np.array([precipitation, precipitationo])
        precip_moy =(np.nansum(tot, axis=0))


        e_precip = multiply_differences(e_precip, [1/precip_moy])*100
        e_num = multiply_differences(e_num, [1/precip_moy])*100
        e_probability = multiply_differences(e_probability, [1/precip_moy])*100
        e_intensity = multiply_differences(e_intensity, [1/precip_moy])*100

        figure_decompo([e_precip, e_num, e_probability, e_intensity], produit+'-'+comparaison+'_20240809_'+s_dom+'_'+debut+fin+'norm', debut, fin,titre+ 'sigma 7')
