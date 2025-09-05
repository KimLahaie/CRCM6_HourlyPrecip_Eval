
###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################

##########################################################################
#                  Importation des modules et libraries                  #
##########################################################################

import parametres

import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib 

from decomposition import calculate_variables
from decomposition import calculate_sensitivity
from decomposition import absolute_error
from decomposition import regime_additive
from decomposition import decomposition_additive
from decomposition import additive_error

##########################################################################
#                               Calculs                                  #
##########################################################################

# Périodes pour chaque saison
periodes = {
    'JJA': np.array(['2016-06', '2016-07', '2016-08', '2017-06', '2017-07', '2017-08'], dtype='datetime64[M]').astype(datetime.datetime),
    'SON': np.array(['2016-09', '2016-10', '2016-11', '2017-09', '2017-10', '2017-11'], dtype='datetime64[M]').astype(datetime.datetime),
    'DJF': np.array(['2016-12', '2016-01', '2016-02', '2017-12', '2017-01', '2017-02'], dtype='datetime64[M]').astype(datetime.datetime),
    'MAM': np.array(['2016-03', '2016-04', '2016-05', '2017-03', '2017-04', '2017-05'], dtype='datetime64[M]').astype(datetime.datetime),
}

simulations = [ 'GEM50_12_C_on_SU_ERA5', 
        'GEM50_2.5_C_on_P3_ERA5', 
        'GEM50_2.5_C_on_SU_ERA5',
        'GEM50_2.5_C_off_P3_ERA5', 
        'GEM50_2.5_I_on_P3_ERA5', 
        'GEM50_2.5_I_off_P3_ERA5',
        'GEM50_2.5_C_off_P3_GEM50_12_SU',
        'GEM50_2.5_C_off_P3_GEM50_12_P3',
        'GEM51_2.5_C_off_P3_GEM51_12_P3']

        
comparaisons = ['GEM50_12_C_on_P3_ERA5',
        'GEM50_12_C_on_P3_ERA5', 
        'GEM50_2.5_C_on_P3_ERA5',  
        'GEM50_2.5_C_on_P3_ERA5', 
        'GEM50_2.5_C_on_P3_ERA5',
        'GEM50_2.5_I_on_P3_ERA5',
        'GEM50_2.5_C_off_P3_ERA5',
        'GEM50_2.5_C_off_P3_GEM50_12_SU',
        'GEM50_2.5_C_off_P3_GEM50_12_P3']


name = np.array(['microphysics: GEM50-12-C-on-SU [ERA5] - GEM50-12-C-on-P3 [ERA5]', 
    'grid spacing: GEM50-2.5-C-on-P3 [ERA5] - GEM50-12-C-on-P3[ERA5]',
    'microphysic: GEM50-2.5-C-on-SU [ERA5] - GEM50-2.5-C-on-P3 [ERA5]',
    'convection: GEM50-2.5-C-off-P3 [ERA5] - GEM50-2.5-C-on-P3 [ERA5]', 
    'LSM: GEM50-2.5-I-on-P3 [ERA5] - GEM50-2.5-C-on-P3[ERA5]', 
    'convection: GEM50-2.5-I-off-P3 [ERA5] - GEM50-2.5-I-on-P3 [ERA5]', 
    'driver: GEM50-2.5-C-off-P3 [GEM50-12-SU] - GEM50-2.5-C-off-P3 [ERA5]',
    'driver: GEM50-2.5-C-off-P3 [GEM50-12-P3] - GEM50-2.5-C-off-P3 [GEM50-12-SU]',
    'GEM version: GEM51-2.5-C-off-P3 [GEM51-12-P3] - GEM50-2.5-C-off-P3 [GEM50-12-P3]'])

colors = np.array(['#000000','#ffbc00','#000000', '#ff0000','#0000ff',  '#ff0000', '#218C21', '#218C21', '#fb3199'])

markers = np.array(["o", "o", "^", "o","o","^","o","^","o"])

s_dom ='USA'
label_size = 20

font = {'family' : 'serif', 'size' : 20}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(2, 2, figsize=[12, 15])
axes = axs.flatten()

for idx, (season, periode) in enumerate(periodes.items()):
    ax = axes[idx]
    ax.set_ylabel('Difference term [mm]')
    ax.set_xlim(0, 10)
    ax.set_xticks([1, 3.5, 6, 8.5])
    ax.set_xticklabels(['$N^s$', '$p^s$', '$I^s$', '$R^s$'])
    ax.set_title(season)
    if idx == 0:  
        ax.set_ylim(-5, 700)
    else:
        ax.set_ylim(-5, 300)
    x = 0
    decalage = np.array([0,0,0,0])

    for w in range(0, len(simulations)):
        produit = simulations[w]
        comparaison = comparaisons[w]
        if produit != ' ':
            data = np.load(parametres.params.DECOMPO_PATH + produit + '/' + produit + '_' + periode[0].strftime("%Y%m") + '_' + s_dom + '.npy')
            data3 = np.load(parametres.params.DECOMPO_PATH + comparaison + '/' +comparaison + '_' + periode[0].strftime("%Y%m") + '_' + s_dom + '.npy')

            for i in range(1, len(periode)):
                data += np.load(parametres.params.DECOMPO_PATH + produit + '/' + produit + '_' + periode[i].strftime("%Y%m") + '_' + s_dom + '.npy')
                data3 += np.load(parametres.params.DECOMPO_PATH + comparaison + '/' + comparaison + '_' + periode[i].strftime("%Y%m") + '_' + s_dom + '.npy')

            precipitation, num, intensity, probability = calculate_variables(data, s_dom)
            precipitationo, numo, intensityo, probabilityo = calculate_variables(data3, s_dom)

            e_precip, e_num, e_probability, e_intensity, residual = calculate_sensitivity(precipitation, num, intensity, probability, precipitationo, numo, intensityo, probabilityo)
           
            tot = np.nansum(np.absolute(e_precip))
            N = np.nansum(np.absolute(e_num))
            P = np.nansum(np.absolute(e_probability))
            I = np.nansum(np.absolute(e_intensity))
            R = np.nansum(np.absolute(residual))

            X = np.array([1, 3.5, 6, 8.5])
            Y = np.array([N, P, I, R])
            ax.scatter(X + decalage, Y, label=name[x], color=colors[x], marker=markers[x], s=160, linewidth=3)
            print('N P I R diff', Y) 
            if w%3 == 1:
                decalage =  np.array([0,0,0,0])
            elif w%3 == 2:
                decalage = decalage + np.array([0.3, 0.3, 0.3,0.3])
            else:
                decalage = decalage - np.array([0.6, 0.6, 0.6, 0.6])

        else:
            ax.scatter(0, 0, label=name[x], color=colors[x])
        x += 1
fig.subplots_adjust(top=0.96, bottom=0.95, wspace=0.2, hspace=0.1)
handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.legend(loc='lower center', bbox_to_anchor=(-0.25, -0.80), ncol=1)

plt.savefig('/transfert/lahaie/fig8.png',bbox_inches='tight')
plt.show()
