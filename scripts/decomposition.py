###########################################################################
#       Évaluation performance - Décomposition intensité-fréquence        #
#            dans le cadre d'un projet de maitrise (E22/A22)              #
#                            par Kim Lahaie                               #
###########################################################################


###########################################################################
#                            Import Modules                              #
###########################################################################
import datetime
import xarray as xr
import numpy as np

import parametres  # Ensure this module is correctly configured

###########################################################################
#                              Definitions                               #
###########################################################################

def read_and_concatenate(start_date, end_date, path):
    """
    Reads and concatenates NetCDF files for precipitation data over a specified period.

    Parameters
    ----------
    start_date : str
        The starting date in the format 'yyyy-mm'.
    end_date : str
        The ending date in the format 'yyyy-mm'.
    path : str
        The directory path where the NetCDF files are stored.

    Returns
    -------
    xr.Dataset
        A concatenated xarray dataset containing precipitation data across the specified period.
    """
    # Generate a range of dates from start_date to end_date
    time_range = np.arange(start_date, end_date, dtype='datetime64[M]').astype(datetime.datetime)

    # Initialize dataset with the first file
    file_path = f"{path}{time_range[0].strftime('%Y%m')}_pr.nc4"
    dataset = xr.open_dataset(file_path)

    # Concatenate datasets from subsequent files
    for i in range(1, len(time_range)):
        file_path = f"{path}{time_range[i].strftime('%Y%m')}_pr.nc4"
        new_dataset = xr.open_dataset(file_path)
        dataset = xr.concat([dataset, new_dataset], dim='time')

    return dataset

def assign_bins(dataset, bin_edges):
    """
    Assigns each data point in a dataset to a bin based on specified thresholds.

    Parameters
    ----------
    dataset : xr.Dataset
        An xarray dataset containing the variable to be binned.
    bin_edges : array-like
        A sequence of bin edges for categorizing data.

    Returns
    -------
    np.ndarray
        An array of integers representing the bin index of each data point.
    """
    # Assume the dataset contains one variable, retrieve its name
    variable_name = list(dataset.keys())[0]
    variable_data = dataset[variable_name].values

    # Assign bin indices using numpy.digitize
    bin_indices = np.digitize(variable_data, bin_edges)

    return bin_indices

def decompose_regime(precip_dataset, tcwv_positions, w_positions, tcwv_bins, w_bins, precip_threshold=parametres.params.SEUIL_PR):
    """
    Decomposes precipitation regimes into statistical arrays based on bin positions.

    Parameters
    ----------
    precip_dataset : xr.Dataset
        Dataset containing precipitation data.
    tcwv_positions : np.ndarray
        Indices representing the bin positions for Total Column Water Vapor (TCWV).
    w_positions : np.ndarray
        Indices representing the bin positions for vertical velocity (w).
    tcwv_bins : array-like
        Bin edges for TCWV.
    w_bins : array-like
        Bin edges for vertical velocity.
    precip_threshold : float, optional
        Threshold for considering precipitation values (default is parametres.params.seuil_pr).

    Returns
    -------
    np.ndarray
        A 3D array containing:
        - The count of data points in each regime.
        - The count of precipitation data points above the threshold in each regime.
        - The sum of precipitation values above the threshold in each regime.
    """
    count_array = np.zeros([len(w_bins), len(tcwv_bins)])
    precip_count_array = np.zeros([len(w_bins), len(tcwv_bins)])
    precip_sum_array = np.zeros([len(w_bins), len(tcwv_bins)])

    for w_idx in range(1, len(w_bins) + 1):
        for tcwv_idx in range(1, len(tcwv_bins) + 1):

            indices = np.where(np.logical_and(w_positions == w_idx, tcwv_positions == tcwv_idx))
            regime_values = precip_dataset.pr.values[indices[0], indices[1], indices[2]]

            precip_indices = np.where(regime_values > precip_threshold)[0]
            regime_precip_values = regime_values[precip_indices]

            count_array[w_idx - 1, tcwv_idx - 1] = len(regime_values)
            precip_count_array[w_idx - 1, tcwv_idx - 1] = len(regime_precip_values)
            precip_sum_array[w_idx - 1, tcwv_idx - 1] = np.sum(regime_precip_values)

    return np.array([count_array, precip_count_array, precip_sum_array])

def calculate_variables(decomposed_data, subdomain, event_threshold=parametres.params.SEUIL_EVENTS):
    """
    Calculates various precipitation-related metrics for a given decomposition.

    Parameters
    ----------
    decomposed_data : np.ndarray
        Decomposed data containing counts and sums of precipitation in different regimes.
    subdomain : str
        Name of the subdomain for determining the size of the domain.
    event_threshold : int, optional
        Minimum number of precipitation events required to include a regime in calculations (default is parametres.params.seuil_events).

    Returns
    -------
    tuple
        A tuple containing arrays for precipitation, event frequency, intensity, and probability.
    """
    domain_size = parametres.params.SUBDOM[subdomain]['size']

    count_array = decomposed_data[0]
    precip_count_array = decomposed_data[1]
    precip_sum_array = decomposed_data[2]

    valid_count_array = np.where(precip_count_array >= event_threshold, count_array, np.nan)
    valid_precip_count_array = np.where(precip_count_array >= event_threshold, precip_count_array, np.nan)
    valid_precip_sum_array = np.where(precip_count_array >= event_threshold, precip_sum_array, np.nan)

    event_frequency = valid_count_array / domain_size
    total_precipitation = valid_precip_sum_array / domain_size
    average_intensity = valid_precip_sum_array / valid_precip_count_array
    probability = valid_precip_count_array / valid_count_array

    return total_precipitation, event_frequency, average_intensity, probability

def calculate_difference(model_quantity, observed_quantity):
    """
    Calculates the difference between modeled and observed quantities.

    Parameters
    ----------
    model_quantity : np.ndarray
        Array of modeled values.
    observed_quantity : np.ndarray
        Array of observed values.

    Returns
    -------
    np.ndarray
        An array of differences.
    """
    total_difference = np.array([model_quantity, -observed_quantity])
    return np.nansum(total_difference, axis=0)

def multiply_differences(total_difference, quantities_to_multiply):
    """
    Multiplies the total difference with additional quantities.

    Parameters
    ----------
    total_difference : np.ndarray
        Array representing the total difference.
    quantities_to_multiply : list of np.ndarray
        List of arrays to multiply with the total difference.

    Returns
    -------
    np.ndarray
        The resulting array after multiplication.
    """
    for quantity in quantities_to_multiply:
        total_difference = total_difference * np.nan_to_num(quantity, nan=1)
    return total_difference

def mask_decomposition(model_quantity, observed_quantity):
    """
    Creates masks based on the presence or absence of NaN values in model and observed quantities.

    Parameters
    ----------
    model_quantity : np.ndarray
        Array of model quantities.
    observed_quantity : np.ndarray
        Array of observed quantities.

    Returns
    -------
    tuple
        A tuple of masks: both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask.
    """
    nan_mask_model = np.isnan(model_quantity)
    nan_mask_observed = np.isnan(observed_quantity)

    both_defined_mask = ~nan_mask_model & ~nan_mask_observed
    both_nan_mask = nan_mask_model & nan_mask_observed
    model_only_mask = ~nan_mask_model & nan_mask_observed
    observed_only_mask = nan_mask_model & ~nan_mask_observed

    return both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask

def calculate_errors_no_mask(precipitation_model, count_model, intensity_model, probability_model,
                              precipitation_observed, count_observed, intensity_observed, probability_observed):
    """
    Calculates errors without applying any masking.

    Parameters
    ----------
    precipitation_model : np.ndarray
        Modeled precipitation values.
    count_model : np.ndarray
        Modeled event counts.
    intensity_model : np.ndarray
        Modeled precipitation intensities.
    probability_model : np.ndarray
        Modeled probabilities.
    precipitation_observed : np.ndarray
        Observed precipitation values.
    count_observed : np.ndarray
        Observed event counts.
    intensity_observed : np.ndarray
        Observed precipitation intensities.
    probability_observed : np.ndarray
        Observed probabilities.

    Returns
    -------
    tuple
        Errors for precipitation, counts, probabilities, intensities, and residuals.
    """
    delta_count = calculate_difference(count_model, count_observed)
    delta_probability = calculate_difference(probability_model, probability_observed)
    delta_intensity = calculate_difference(intensity_model, intensity_observed)
    precipitation_error = calculate_difference(precipitation_model, precipitation_observed)
    
    count_error = multiply_differences(delta_count, [intensity_observed, probability_observed])
    probability_error = multiply_differences(delta_probability, [count_observed, intensity_observed])
    intensity_error = multiply_differences(delta_intensity, [count_observed, probability_observed])

    residual_error = (multiply_differences(delta_count, [delta_probability, intensity_observed]) +
                      multiply_differences(delta_probability, [delta_intensity, count_observed]) +
                      multiply_differences(delta_intensity, [delta_count, probability_observed]) +
                      multiply_differences(delta_probability, [delta_intensity, delta_count]))
    
    return precipitation_error, count_error, probability_error, intensity_error, residual_error


def calculate_sensitivity_no_mask(precipitation_model, count_model, intensity_model, probability_model,
                              precipitation_changed, count_changed, intensity_changed, probability_changed):
    """
    Calculates difference to assess sentivity  without applying any masking.

    Parameters
    ----------
    precipitation_model : np.ndarray
        Modeled precipitation values.
    count_model : np.ndarray
        Modeled event counts.
    intensity_model : np.ndarray
        Modeled precipitation intensities.
    probability_model : np.ndarray
        Modeled probabilities.
    precipitation_changed : np.ndarray
        Modeled precipitation values with a unique change in configuration.
    count_changed : np.ndarray
        Modeled event counts with a unique change in configuration.
    intensity_changed : np.ndarray
        Modeled precipitation intensities with a unique change in configuration.
    probability_observed : np.ndarray
        Modeled probabilities with a unique change in configuration.

    Returns
    -------
    tuple
        Difference for precipitation, counts, probabilities, intensities, and residuals.
    """

    delta_count = calculate_difference(count_model, count_changed)
    delta_probability = calculate_difference(probability_model, probability_changed)
    delta_intensity = calculate_difference(intensity_model, intensity_changed)
    precipitation_error = calculate_difference(precipitation_model, precipitation_changed)

    count_avg = (count_model + count_changed)/2
    intensity_avg = (intensity_model + intensity_changed)/2
    probability_avg = (probability_model + probability_changed)/2

    count_error = multiply_differences(delta_count, [intensity_avg, probability_avg])
    probability_error = multiply_differences(delta_probability, [count_avg, intensity_avg])
    intensity_error = multiply_differences(delta_intensity, [count_avg, probability_avg])

    residual_error = (multiply_differences(delta_count, [delta_probability, intensity_avg]) +
                      multiply_differences(delta_probability, [delta_intensity, count_avg]) +
                      multiply_differences(delta_intensity, [delta_count, probability_avg]) +
                      multiply_differences(delta_probability, [delta_intensity, delta_count])*(1/2))

    return precipitation_error, count_error, probability_error, intensity_error, residual_error

def apply_error_masking(precipitation_error, count_error, probability_error, intensity_error, residual_error,
                        both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask):
    """
    Applies masking to errors based on the presence or absence of values in model and observed data.

    Parameters
    ----------
    precipitation_error : np.ndarray
        Precipitation error array.
    count_error : np.ndarray
        Count error array.
    probability_error : np.ndarray
        Probability error array.
    intensity_error : np.ndarray
        Intensity error array.
    residual_error : np.ndarray
        Residual error array.
    both_defined_mask : np.ndarray
        Mask where both model and observed values are defined.
    both_nan_mask : np.ndarray
        Mask where both model and observed values are NaN.
    model_only_mask : np.ndarray
        Mask where only model values are defined.
    observed_only_mask : np.ndarray
        Mask where only observed values are defined.

    Returns
    -------
    tuple
        Masked errors for precipitation, counts, probabilities, intensities, and residuals.
    """
    precipitation_error[both_nan_mask] = np.nan
    count_error[both_nan_mask] = np.nan
    count_error[model_only_mask] = precipitation_error[model_only_mask]
    count_error[observed_only_mask] = precipitation_error[observed_only_mask]

    probability_error[both_nan_mask] = np.nan
    probability_error[model_only_mask] = 0
    probability_error[observed_only_mask] = 0

    intensity_error[both_nan_mask] = np.nan
    intensity_error[model_only_mask] = 0
    intensity_error[observed_only_mask] = 0

    residual_error[both_nan_mask] = np.nan
    residual_error[model_only_mask] = 0
    residual_error[observed_only_mask] = 0
   
    return precipitation_error, count_error, probability_error, intensity_error, residual_error

def calculate_errors(precipitation_model, count_model, intensity_model, probability_model,
                     precipitation_observed, count_observed, intensity_observed, probability_observed):
    """
    Calculates and masks errors for precipitation, counts, probabilities, and intensities.

    Parameters
    ----------
    precipitation_model : np.ndarray
        Modeled precipitation values.
    count_model : np.ndarray
        Modeled event counts.
    intensity_model : np.ndarray
        Modeled precipitation intensities.
    probability_model : np.ndarray
        Modeled probabilities.
    precipitation_observed : np.ndarray
        Observed precipitation values.
    count_observed : np.ndarray
        Observed event counts.
    intensity_observed : np.ndarray
        Observed precipitation intensities.
    probability_observed : np.ndarray
        Observed probabilities.

    Returns
    -------
    tuple
        Masked errors for precipitation, counts, probabilities, intensities, and residuals.
    """
    both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask = mask_decomposition(count_model, count_observed)
    
    precipitation_error, count_error, probability_error, intensity_error, residual_error = calculate_errors_no_mask(
        precipitation_model, count_model, intensity_model, probability_model,
        precipitation_observed, count_observed, intensity_observed, probability_observed
    )

    return apply_error_masking(precipitation_error, count_error, probability_error, intensity_error, residual_error,
                               both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask)


def calculate_sensitivity(precipitation_model, count_model, intensity_model, probability_model,
                     precipitation_changed, count_changed, intensity_changed, probability_changed):
    """
    Calculates and masks differences for precipitation, counts, probabilities, and intensities.

    Parameters
    ----------
    precipitation_model : np.ndarray
        Modeled precipitation values.
    count_model : np.ndarray
        Modeled event counts.
    intensity_model : np.ndarray
        Modeled precipitation intensities.
    probability_model : np.ndarray
        Modeled probabilities.
    precipitation_changed : np.ndarray
        Modeled precipitation values with a unique configuration change.
    count_changed : np.ndarray
        Modeled event counts with a unique configuration change.
    intensity_changed : np.ndarray
        Modeled precipitation intensities with a unique configuration change.
    probability_changed : np.ndarray
        Modeled probabilities with a unique change in configuration.

    Returns
    -------
    tuple
        Masked difference for precipitation, counts, probabilities, intensities, and residuals.
    """
    both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask = mask_decomposition(count_model, count_changed)
    
    precipitation_error, count_error, probability_error, intensity_error, residual_error = calculate_errors_no_mask(
        precipitation_model, count_model, intensity_model, probability_model,
        precipitation_changed, count_changed, intensity_changed, probability_changed
    )

    return apply_error_masking(precipitation_error, count_error, probability_error, intensity_error, residual_error,
                               both_defined_mask, both_nan_mask, model_only_mask, observed_only_mask)


