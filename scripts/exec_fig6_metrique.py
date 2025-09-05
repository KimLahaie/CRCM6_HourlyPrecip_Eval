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

from decomposition import calculate_variables
from decomposition import calculate_errors
from decomposition import absolute_error
from decomposition import regime_additive
from decomposition import decomposition_additive
from decomposition import additive_error

##########################################################################
#                   Configurations choices                               #
##########################################################################


periodes = {
    'JJA': np.array(['2016-06', '2016-07', '2016-08', '2017-06', '2017-07', '2017-08'], dtype='datetime64[M]').astype(datetime.datetime),
    'SON': np.array(['2016-09', '2016-10', '2016-11', '2017-09', '2017-10', '2017-11'], dtype='datetime64[M]').astype(datetime.datetime),
    'DJF': np.array(['2016-12', '2016-01', '2016-02', '2017-12', '2017-01', '2017-02'], dtype='datetime64[M]').astype(datetime.datetime),
    'MAM': np.array(['2016-03', '2016-04', '2016-05', '2017-03', '2017-04', '2017-05'], dtype='datetime64[M]').astype(datetime.datetime),
}

simulations = ['GEM50_2.5_I_on_P3_ERA5', 'GEM50_2.5_I_off_P3_ERA5', 'GEM50_2.5_C_on_P3_ERA5', 'GEM50_2.5_C_off_P3_ERA5', 'GEM50_2.5_C_on_SU_ERA5', 'GEM50_2.5_C_off_P3_GEM50_12_P3', 'GEM50_2.5_C_off_P3_GEM50_12_SU', 'GEM51_2.5_C_off_P3_GEM51_12_P3',' ', 'GEM50_12_C_on_P3_ERA5', 'GEM50_12_C_on_SU_ERA5',' ', 'REANALYSIS_ERA5', 'IMERG_V7', 'NEXRAD_STAGE_IV', 'IMERG_V6']

name = np.array(['GEM50-2.5-I-on-P3 [ERA5]', 'GEM50-2.5-I-off-P3 [ERA5]', 'GEM50-2.5-C-on-P3 [ERA5]', 'GEM50-2.5-C-off-P3 [ERA5]', 'GEM50-2.5-C-on-SU [ERA5]', 'GEM50-2.5-C-off-P3 [GEM50-12-P3]', 'GEM50-2.5-C-off-P3 [GEM50-12-SU]', 'GEM51-2.5-C-off-P3 [GEM51-12-P3]','- - - - - - - - - - - - - - - - - - - - - - - - - - ', 'GEM50-12-C-on-P3 [ERA5]', 'GEM50-12-C-on-SU [ERA5]','- - - - - - - - - - - - - - - - - - - - - - - - - - ',  'ERA5', 'IMERG V7', 'STAGE IV', 'IMERG V6'])

colors = np.array(["#f79256","#f79256","#f79256","#f79256","#f79256","#c66122","#f9b27c","#764425","#ffffff", "#0f80aa", "#0f80aa","#ffffff", '#569a12',"#0a0a0a",'#0a0a0a','#0a0a0a'])

markers = np.array(["o", "o", "^", "^", "v","s", "p","d",".", "s", "p",".",  "D", "*", "X","P" ])

filled = np.array(["none","#f79256","none","#f79256","none","#c66122","#f9b27c","#764425","none", "none","none","none", '#569a12', '#0a0a0a', '#0a0a0a', '#0a0a0a'])

label_size = 20
import matplotlib as matplotlib
font = {'family' : 'serif',
        'size'   : 20}
matplotlib.rc('font', **font)

s_dom='USA'

fig, axs = plt.subplots(2, 2, figsize=[12, 15])
axes = axs.flatten()

for idx, (season, periode) in enumerate(periodes.items()):
    ax = axes[idx]
    ax.set_xlim([-8,175])
    ax.set_ylim([-40,1800])
    ax.set_ylabel('$PT_\\epsilon^{AAE}$ [mm]')
    ax.set_xlabel('$PT_\\epsilon^{AE}$ [mm]')#, labelpad=-20)
    ax.set_title(season)
    x = 0

    for produit in simulations:
        if produit != ' ':
            data = np.load(parametres.params.DECOMPO_PATH + produit + '/' + produit + '_' + periode[0].strftime("%Y%m") + '_' + s_dom + '.npy')
            data3 = np.load(parametres.params.DECOMPO_PATH + 'IMERG_V7/IMERG_V7_' + periode[0].strftime("%Y%m") + '_' + s_dom + '.npy')

            for i in range(1, len(periode)):
                data += np.load(parametres.params.DECOMPO_PATH + produit + '/' + produit + '_' + periode[i].strftime("%Y%m") + '_' + s_dom + '.npy')
                data3 += np.load(parametres.params.DECOMPO_PATH + 'IMERG_V7/IMERG_V7_' + periode[i].strftime("%Y%m") + '_' + s_dom + '.npy')

            precipitation, num, intensity, probability = calculate_variables(data, s_dom)
            precipitationo, numo, intensityo, probabilityo = calculate_variables(data3, s_dom)

            e_precip, e_num, e_probability, e_intensity, residual = calculate_errors(precipitation, num, intensity, probability, precipitationo, numo, intensityo, probabilityo)

            AE = absolute_error(e_num, e_probability, e_intensity, residual)
            R_AAE = regime_additive(e_num, e_probability, e_intensity, residual)
            EFI_AAE = decomposition_additive(e_num, e_probability, e_intensity, residual)
            AAE = additive_error(e_num, e_probability, e_intensity, residual)

            ax.scatter(AE, AAE, label=name[x], color=colors[x], marker=markers[x], s=160, facecolors=filled[x], linewidths=3)
        else:
            ax.scatter(0, 0, label=name[x], color=colors[x])
        x += 1
fig.subplots_adjust(top=0.96, bottom=0.95, wspace=0.2, hspace=0.1)
handles, labels = ax.get_legend_handles_labels()
fig.tight_layout()
plt.legend(loc='lower center', bbox_to_anchor=(-0.35, -0.80), ncol=2)
plt.savefig('/transfert/lahaie/fig6.png', bbox_inches='tight')
plt.show()
