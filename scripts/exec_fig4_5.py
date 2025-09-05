###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################


##########################################################################
#                  Importation des modules et libraries                  #
##########################################################################

import parametres

from decomposition import calculate_variables
from decomposition import calculate_errors

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import math
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
#########################################################################
#
#########################################################################

font = {'family' : 'serif',
        'size'   : 24}
mpl.rc('font', **font)

 
def figure_decompo(values, name, debut, fin,titre, precip_Array, intensity_Array, prob_Array):
    
    cmap_data = ["#45220d","#934616","#e07d40","#f0bea0","#FFFFFF","#80D9E5","#00b2ca","#16679a", "#113055"]

    yticks = [-3, -2,-1, 0,1, 2]
    xticks = [0, 25,50,75]

    cmap_precip = LinearSegmentedColormap.from_list('custom_cmap', cmap_data)
    bounds = [-1000,-100, -10, -1, -0.1,-0.00000004, 0.00000004, 0.1, 1, 10,100,1000]
    norm = BoundaryNorm(bounds, cmap_precip.N)
    print(norm)
    bounds_label = ['-10$^{2}$','-10$^{1}$', '-10$^{0}$', '-10$^{-1}$','0', '10$^{-1}$', '10$^{0}$', '10$^{1}$','10$^{2}$']
    fig = plt.figure(figsize=(11, 11))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    cax = fig.add_subplot(gs[:, 2]) 

    bins_tcwv = parametres.params.BINS_TCWV
    bins_w = parametres.params.BINS_W

    cmap = mpl.cm.get_cmap(cmap_precip)
    cmap.set_bad(color='gray')

    p1 = ax1.pcolormesh(bins_tcwv, bins_w, values[0], cmap=cmap, norm=norm)
    ax1.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax1.set_xlabel('W [mm]')
    ax1.set_yticks(yticks)
    ax1.set_xticks(xticks)
    ax1.set_ylim(-3.1, 1.3)
    ax1.set_title('a)', loc='left', pad=10)

    true_indices = np.where(precip_Array == 3)
    x_true = true_indices[1]*5 + 2.5  
    y_true = true_indices[0]*0.2 + 0.1-3.1  
    ax1.scatter(x_true, y_true, marker='+', color='black', s=48)
    true_indices = np.where(precip_Array == -3)
    x_true = true_indices[1]*5 + 2.5  
    y_true = true_indices[0]*0.2 + 0.1-3.1
    ax1.scatter(x_true, y_true, marker='_', color='black', s=48)

    p2 = ax2.pcolormesh(bins_tcwv, bins_w, values[1], cmap = cmap, norm=norm)
    ax2.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax2.set_xlabel('W [mm]')
    ax2.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_ylim(-3.1, 1.3)
    ax2.set_title('b)', loc='left', pad=10)

    p3 = ax3.pcolormesh(bins_tcwv, bins_w, values[2], cmap = cmap, norm=norm)
    ax3.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax3.set_xlabel('W [mm]')
    ax3.set_yticks(yticks)
    ax3.set_xticks(xticks)
    ax3.set_ylim(-3.1, 1.3)
    ax3.set_title('c)', loc='left', pad =10)
    
    true_indices = np.where(prob_Array == 3)
    x_true = true_indices[1]*5 + 2.5
    y_true = true_indices[0]*0.2 + 0.1-3.1  
    ax3.scatter(x_true, y_true, marker='+', color='black', s=48)
    true_indices = np.where(prob_Array == -3)
    x_true = true_indices[1]*5 + 2.5  
    y_true = true_indices[0]*0.2 + 0.1-3.1
    ax3.scatter(x_true, y_true, marker='_', color='black', s=48)

    p4 = ax4.pcolormesh(bins_tcwv, bins_w, values[3], cmap = cmap, norm=norm)
    ax4.set_ylabel('$\omega$ [Pa s$^{-1}$]')
    ax4.set_xlabel('W [mm]')
    ax4.set_yticks(yticks)
    ax4.set_xticks(xticks)
    ax4.set_ylim(-3.1, 1.3)
    ax4.set_title('d)', loc='left', pad = 10)
    
    true_indices = np.where(intensity_Array == 3)
    x_true = true_indices[1]*5 + 2.5  
    y_true = true_indices[0]*0.2 + 0.1-3.1  
    ax4.scatter(x_true, y_true, marker="+", color='black', s=48, edgecolors=None)
    true_indices = np.where(intensity_Array == -3)
    x_true = true_indices[1]*5 + 2.5  
    y_true = true_indices[0]*0.2 + 0.1-3.1
    
    nom = name.split('USA', 1)
    fig.suptitle(titre)
    label_size=24
    ax1.text(0.75,0.05 , '$\widetilde{PT}^{\epsilon}$' ,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax1.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax2.text(0.79,0.05 , '$N^{\epsilon}$',
        horizontalalignment='left',
        verticalalignment='bottom', 
        transform=ax2.transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax3.text(0.8,0.05 , '$p^{\epsilon}$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax3.transAxes,bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)
    ax4.text(0.83,0.05 , '$I^{\epsilon}$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax4.transAxes, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, size = label_size)

    fig.subplots_adjust(left=0.11, right=0.85, top=0.86, bottom=0.09, wspace=0.60, hspace=0.35)

    for ax in fig.get_axes():
        if ax not in [ax1, ax2, ax3, ax4]:
            fig.delaxes(ax)  

    cax = fig.add_axes([0.8, 0.1, 0.03, 0.8])

    heights = [0.1, 0.1, 0.1, 0.1, 0.01,0.1, 0.1, 0.1, 0.1]
    tick_positions = [0, 0.1, 0.2, 0.3, 0.405, 0.51, 0.61, 0.71,0.81]
    tick_labels = ['-10$^3$', '-10$^2$', '-10$^1$', '-10$^{-1}$', '0', '10$^{-1}$', '10$^1$', '10$^2$', '10$^3$']
    start = 0

    for color, height in zip(cmap_data, heights):
        rect = mpatches.Rectangle((0, start), 1, height, color=color, ec='none')
        cax.add_patch(rect)
        start += height

    cax.set_xlim(0, 1)
    cax.set_ylim(0, start)  

    cax.set_yticks(tick_positions)
    cax.set_yticklabels(tick_labels)
    cax.yaxis.set_tick_params(width=1, labelright=True, labelleft=False, right=True, left=False)  
    cax.xaxis.set_tick_params(width=0, labelright=False, labelleft=False)
    cax.set_ylabel('[mm]', rotation=-90, labelpad=-75, va='bottom')
    plt.savefig('/transfert/lahaie/'+name+'.png')
    return None

#########################################################################
#  Initialisation pour tracer figures decompo et decompo-erreur         #
#########################################################################

debut='2016-01'
fin='2018-01'

produits = ['GEM50_12_C_on_SU_ERA5', 'GEM50_2.5_C_on_P3_ERA5']
comparaisons = ['IMERG_V7', 'IMERG_V7']
comparaisons2 = ['IMERG_V6', 'IMERG_V6']
comparaisons3 =  ['NEXRAD_STAGE_IV', 'NEXRAD_STAGE_IV']
titres = ['GEM50-12-C-on-SU [ERA5] -\n IMERG V7', 'GEM50-2.5-C-on-P3 [ERA5] -\n IMERG V7']

s_dom = 'USA' 
for i in range(len(produits)):
        produit = produits[i]
        titre = titres[i]
        comparaison = comparaisons[i]
        comparaison2 = comparaisons2[i]
        comparaison3 = comparaisons3[i]
        periode  = np.arange(debut, fin, dtype='datetime64[M]').astype(datetime.datetime)

        data = np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        data2 = np.load(parametres.params.DECOMPO_PATH+comparaison+'/'+comparaison+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        data3 = np.load(parametres.params.DECOMPO_PATH+comparaison2+'/'+comparaison2+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        data4 = np.load(parametres.params.DECOMPO_PATH+comparaison3+'/'+comparaison3+'_'+periode[0].strftime("%Y%m")+'_'+s_dom+'.npy')
        for i in range(1, len(periode)):
            data += np.load(parametres.params.DECOMPO_PATH+produit+'/'+produit+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
            data2 += np.load(parametres.params.DECOMPO_PATH+comparaison+'/'+comparaison+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
            data3 += np.load(parametres.params.DECOMPO_PATH+comparaison2+'/'+comparaison2+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')
            data4 += np.load(parametres.params.DECOMPO_PATH+comparaison3+'/'+comparaison3+'_'+periode[i].strftime("%Y%m")+'_'+s_dom+'.npy')

        precipitation, num, intensity, probability = calculate_variables(data, s_dom)
        precipitationo, numo, intensityo, probabilityo = calculate_variables(data2, s_dom)
        precipitationo2, numo2, intensityo2, probabilityo2 = calculate_variables(data3, s_dom)
        precipitationo3, numo3, intensityo3, probabilityo3 = calculate_variables(data4, s_dom)
        e_precip, e_num, e_probability, e_intensity, residual = calculate_errors(precipitation, num, intensity, probability, precipitationo,numo, intensityo, probabilityo)
        e_precip2, e_num2, e_probability2, e_intensity2, residual2 = calculate_errors(precipitation, num, intensity, probability, precipitationo2,numo2, intensityo2, probabilityo2)

        e_precip3, e_num3, e_probability3, e_intensity3, residual3 = calculate_errors(precipitation, num, intensity, probability, precipitationo3,numo3, intensityo3, probabilityo3)

        signe_prob = e_probability3/np.abs(e_probability3) + e_probability2/np.abs(e_probability2) + e_probability/np.abs(e_probability)
        signe_intensity = e_intensity3/np.abs(e_intensity3) + e_intensity2/np.abs(e_intensity2) + e_intensity/np.abs(e_intensity)
        signe_precip =e_precip3/np.abs(e_precip3) +  e_precip2/np.abs(e_precip2) + e_precip/np.abs(e_precip)
        figure_decompo([e_precip, e_num, e_probability, e_intensity], produit+'-'+comparaison+s_dom, debut, fin,titre,  signe_precip, signe_intensity, signe_prob)
