# CRCM6-GEM5 – Performance and Sensitivity Assessment of Hourly Precipitation

This repository contains the code used for the analysis presented in the manuscript:

> **Lahaie, K., Di Luca, A., Roberge, F., Paquin-Ricard, D. (2025).**  
> *Canadian Regional Climate Model Performance and Sensitivity Assessment for Hourly Precipitation*.  
> *Journal of Advances in Modeling Earth Systems (JAMES)*. [DOI to be added]

The processed data from all CRCM6-GEM5 numerical experiments, all observations and reanalysis data used are publicly available in the Borealis data repository at https://doi.org/10.5683/SP3/M3SQ8P.

---

## Project Overview
# GENERAL INFORMATION

**1. Authors Information**  
Kim Lahaie (a), Alejandro Di Luca (a), Frédéric Roberge (b), Danahé Paquin-Ricard (b)

**2. Affiliation**  
a) Centre pour l’Étude et la Simulation du Climat à l’Échelle Régionale (ESCER) | Département des sciences de la Terre et de l’atmosphère, Université du Québec à Montréal (UQAM), Montréal, Canada.  
b) Environnement et Changement climatique Canada (ECCC), Canada.

**3. Funding and Computational Resources**  
This research was conducted as part of the project “Simulation et analyse du climat à haute résolution” with financial support from the Government of Québec.  
This research was enabled in part by support provided by **Calcul Québec** (https://www.calculquebec.ca) and the **Digital Research Alliance of Canada** (https://alliancecan.ca).

A. Di Luca (RGPIN-2020-05631) and K. Lahaie (575955-2022) were funded by the **Natural Sciences and Engineering Research Council of Canada (NSERC)**.

**4. Summary
This project evaluates the performance and sensitivity of different configurations of the CRCM6-GEM5 regional climate model in simulating **hourly precipitation** over **northeastern North America**. The evaluation uses a decomposition method to link precipitation errors to their **environmental**, **frequency**, and **intensity** components.

The analysis combines:
- **CRCM6-GEM5 simulations** at 2.5 km and 12 km resolutions
- **ERA5 reanalysis** fields (vertical velocity and water vapor)
- **IMERG V6/V7** and **Stage IV** precipitation observations
The simulations, observations and reanalysis data are available at https://doi.org/10.5683/SP3/M3SQ8P

Precipitation errors are analyzed using the **Environmentally Conditioned Intensity-Frequency (ECIF)** decomposition method introduced by Di Luca, A., de Elía, R., Bador, M., & Argüeso, D. (2020). Separating the Influences of Physically Different Processes on Simulated Precipitation Errors at the Regional Scale. Journal of Advances in Modeling Earth Systems, 12(11), e2020MS002447. https://doi.org/10.1029/2020MS002447.

# SHARING/ACCESS INFORMATION

1-Licenses/Restrictions placed on the data: These data are available under a CC BY 4.0 license https://creativecommons.org/licenses/by/4.0/

# FILE OVERVIEW

- **parametres.py**  
  Defines the parameters to be specified by the user.

- **sauvegarde\_decompo\_mensuelle.py**  
  Executes the most computationally intensive step, producing a 3D array with:  
  - the count of data points in each regime,  
  - the count of precipitation data points above the threshold in each regime,  
  - the sum of precipitation values above the threshold in each regime.  

  These statistical values allow for quick calculation of **N**, **I**, **P**, and **PR**.  
  Results are saved to avoid repeating long calculations.

- **decomposition.py**  
  Contains all core functions used in the analysis.

- **exec\_fig3\_decompo.py** 
- **exec\_fig4\_5.py** 
- **exec\_fig6\_metrique.py**
- **exec\_fig7\_metrique.py**
- **exec\_fig8\_metrique.py**  
  Scripts for generating the figures.
