"""
model.py

This module contains functions for predicting lensing probability density functions (PDF) using 
pre-trained XGBoost models. It includes input validation to ensure that the necessary parameters 
are provided for making predictions.

Functions:
----------
- load_model(model_name: str) -> xgb.Booster:
    Loads a pre-trained XGBoost model.
    
- load_training_data() -> pd.DataFrame:
    Loads the training data.

- load_pca() -> PCA:
    Loads the PCA (Principal Component Analysis) transformation model.

- load_scaling() -> StandardScaler:
    Loads the input data scaling model.

- check_parameters(Om: float, h: float, w: float, s8: float) -> bool:
    Checks if the input cosmological parameters fall within specified ranges.

- predict_pdf(Om: float, h: float, w: float, s8: float, z: float) -> tuple:
    Predicts the probability density function (PDF) based on input cosmological parameters.

- predict_sigma(Om: float, h: float, w: float, s8: float, z: float) -> numpy.ndarray:
    Predicts the sigma (standard deviation) for the PDF based on input cosmological parameters.

- predict_mean(Om: float, h: float, w: float, s8: float, z: float) -> numpy.ndarray:
    Predicts the mean for the PDF based on input cosmological parameters.

- transform_input(Om: float, h: float, w: float, s8: float, z: float) -> numpy.ndarray:
    Validates and transforms the input parameters into a standardized format for model prediction.
"""
import numpy as np
import pkg_resources
import xgboost as xgb
import pandas as pd
import pickle


def enforce_monotonicity(row, window_size=4):
    """
    Enforces monotonicity on a given 1D array (`row`) by ensuring an increasing trend up to the 
    peak value and a decreasing trend afterward. Any violations of monotonicity are handled 
    by setting elements to zero, starting from the point of violation.

    Parameters:
    -----------
    row : array-like
        The 1D array on which monotonicity is enforced. 
    window_size : int, optional
        The size of the window to use for checking monotonicity. Defaults to 4.

    Returns:
    --------
    numpy.ndarray
        The modified array with enforced monotonicity.

    Notes:
    ------
    - The function identifies the peak in `row` (the maximum value) and ensures values 
      up to this peak are monotonically increasing.
    - After the peak, the function enforces a monotonically decreasing trend.
    - If a monotonic trend is not detected within the specified `window_size`, the 
      function zeroes out values starting from the violation point backward (for the 
      increasing section) or forward (for the decreasing section).
    """
    row = np.array(row, copy=True)
    # Find the index of the peak point
    peak_index = np.argmax(row)

    # Enforce monotonic increase up to the peak
    for i in range(peak_index - 1, -1, -1):
        # Ensure we are not going out of bounds for the window
        if i <= window_size - 1:
            # If we can't form a complete window, only last points
            trend = row[i + 1] > row[i]
        else:
            # Compare the current window with the previous window we use any to avoid tiny fluctuations
            trend = np.any(row[i - window_size + 1:i + 1] >= row[i - window_size:i])
        
        if not trend: 
            row[:i + 1] = 0  # Set all previous values to 0
            break

    # Enforce monotonic decrease after the peak
    for i in range(peak_index, len(row) - 1):
        if (i + window_size) >= len(row):
            trend = row[i] <= row[i-1]
        else:
            trend = np.any(row[i:i + window_size] >= row[i + 1:i + 1 + window_size])
        
        if not trend: 
            row[i + 1:] = 0 
            break
    return row


def smooth_pdf_fourier(pdf, cutoff=10, x=None):
    """
    Smooth a 1D PDF using Fourier transform with a given cutoff frequency.

    Parameters:
    -----------
    pdf : np.ndarray
        1D array of the PDF to smooth (expected length: 5000).
    cutoff : float
        Frequency cutoff for smoothing (default: 10).
    x : np.ndarray or None
        Optional array of x values corresponding to the PDF.
        If None, a default linspace from -2 to 8 is used.

    Returns:
    --------
    pdf_smooth : np.ndarray
        Smoothed PDF (real values only).
    """
    if x is None:
        x = np.linspace(-2, 8, len(pdf))
    
    # Compute FFT
    fft = np.fft.fft(pdf)
    freqs = np.fft.fftfreq(len(pdf), d=(x[1] - x[0]))
    
    # Zero out frequencies above cutoff
    fft[np.abs(freqs) > cutoff] = 0
    
    # Inverse FFT to get smoothed PDF
    pdf_smooth = np.fft.ifft(fft).real
    return pdf_smooth


def load_model(model_name: str) -> xgb.Booster:
    """
    Load a pre-trained XGBoost model.

    Parameters:
    -----------
    model_name : str
        Name of the model to load.

    Returns:
    --------
    xgb.Booster
        Loaded XGBoost model.

    Raises:
    -------
    FileNotFoundError
        If the model file is not found.
    Exception
        If any other error occurs while loading the model.
    """
    model_path = pkg_resources.resource_filename('ace_lensing', f'models/{model_name}.xgb')
    try:
        model = xgb.Booster(model_file=model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading model: {e}")


def load_training_data() -> pd.DataFrame:
    """
    Load the training data by combining separate Parquet files for different columns.
    Returns:
    --------
    pd.DataFrame
        Training data in pandas DataFrame format with all necessary columns.
    Raises:
    -------
    FileNotFoundError
        If any of the training data files are not found.
    Exception
        If any other error occurs while loading the training data.
    """
    base_path = 'data/training_set_{}.parquet'
    try:
        train_mu_vec = pd.read_parquet(pkg_resources.resource_filename('ace_lensing', base_path.format('mu_vec')))
        train_pdf = pd.read_parquet(pkg_resources.resource_filename('ace_lensing', base_path.format('pdf')))
        train_error = pd.read_parquet(pkg_resources.resource_filename('ace_lensing', base_path.format('error')))
        train_cosmo = pd.read_parquet(pkg_resources.resource_filename('ace_lensing', base_path.format('cosmo')))
        train_statistics = pd.read_parquet(pkg_resources.resource_filename('ace_lensing', base_path.format('statistics')))
        # Concatenate the DataFrames along the columns
        df = pd.concat([train_mu_vec, train_pdf['pdf'], train_error['poisson'],
                        train_cosmo[['Om', 'h', 'w', 's8', 'z']],
                        train_statistics[['mean', 'var', '3th', '4th', '5th', '6th', '7th', '8th', '9th', '10th']]],
                       axis=1)
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Training data file not found: {e.filename}")
    except Exception as e:
        raise Exception(f"An error occurred while loading training data: {e}")


def load_pca():
    """
    Load PCA (Principal Component Analysis) transformation model.

    Returns:
    --------
    PCA
        Loaded PCA model.

    Raises:
    -------
    FileNotFoundError
        If the PCA model file is not found.
    Exception
        If any other error occurs while loading the PCA model.
    """
    pca_path = pkg_resources.resource_filename('ace_lensing', 'models/pca.pkl')
    try:
        with open(pca_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"PCA model file not found: {pca_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading PCA model: {e}")

    
def load_scaling():
    """
    Load input data scaling model.

    Returns:
    --------
    scale
        The scaling model.

    Raises:
    -------
    FileNotFoundError
        If the scaling model file is not found.
    Exception
        If any other error occurs while loading the scaling model.
    """
    scale_path = pkg_resources.resource_filename('ace_lensing', 'models/scale.pkl')
    try:
        with open(scale_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaling model file not found: {scale_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading scaling model: {e}")


def check_parameters(Om: float, h: float, w: float, s8: float, z: float) -> bool:
    """
    Checks if the input cosmological parameters fall within specified ranges for 
    predicting the PDF.

    Parameters:
    -----------
    Om : float
        Matter density parameter, should be between 0.2 and 0.4.
    h : float
        Hubble parameter, should be between 60 and 80.
    w : float
        Dark energy equation of state parameter, should be between -1.5 and -0.7.
    s8 : float
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc, 
        should be between 0.7 and 0.9.
    z : float
        Redshift value, should be between 0.2 and 6.

    Returns:
    --------
    bool
        True if all parameters are within their specified ranges.

    Raises:
    -------
    ValueError
        If any parameter is outside the defined range, an error is raised 
        indicating which parameter is invalid.
    """
    if not (0.2 <= Om <= 0.4):
        raise ValueError(f"Om value {Om} is out of range (0.2 - 0.4)")
    if not (0.6 <= h <= 0.8):
        raise ValueError(f"h value {h} is out of range (0.6 - 0.8)")
    if not (-1.5 <= w <= -0.7):
        raise ValueError(f"w value {w} is out of range (-1.5 - -0.7)")
    if not (0.7 <= s8 <= 0.9):
        raise ValueError(f"s8 value {s8} is out of range (0.7 - 0.9)")
    if not (0.2 <= z <= 6):
        raise ValueError(f"z value {z} is out of range (0.2 - 6)")
    
    return True


def predict_pdf(Om: float, h: float, w: float,  s8: float, z: float, verbose=True):
    """
    Predict the probability density function (PDF) for given cosmological parameters using
    pre-trained XGBoost models and inverse PCA reconstruction, followed by Fourier smoothing
    and normalization.

    Parameters
    ----------
    Om : float
        Matter density parameter, typically in the range 0.2 to 0.4.
    h : float
        Dimensionless Hubble parameter, typically in the range 0.6 to 0.8.
    w : float
        Dark energy equation of state parameter, typically in the range -1.5 to -0.7.
    s8 : float
        Standard deviation of matter density fluctuations, typically in the range 0.7 to 0.9.
    z : float
        Redshift value (non-negative float).
    verbose : bool, optional
        If True, prints the input parameters and diagnostic information. Default is True.

    Returns
    -------
    mu_vec_prime : numpy.ndarray of shape (5000,)
        Standardized and scaled mu vector corresponding to the PDF values.
    pdf_prime_normalized : numpy.ndarray of shape (5000,)
        Normalized PDF corresponding to `mu_vec_prime`. The values are non-negative and sum to 1
        under the integration over `mu_vec_prime`.

    Notes
    -----
    - The function performs the following steps:
        1. Checks input parameters for validity.
        2. Transforms inputs into model features.
        3. Uses pre-trained XGBoost models to predict PCA components of the PDF.
        4. Reconstructs the PDF via inverse PCA transformation.
        5. Enforces monotonicity and smooths the PDF using Fourier transform with a cutoff of 10.
        6. Uses  pre-trained XGBoost models to predict mean and sigma.
        7. Unstandardized the PDF.
        8. Forces mean to be 1.
        9. Normalizes the resulting PDF.
        
    - The resulting `mu_vec_prime` and `pdf_prime_normalized` can be used for plotting or further analysis.
    """

    # Count the number of parameters provided
    params = [Om, h, w, s8, z]
    num_provided = sum(p is not None for p in params)

    check_parameters(*params)

    if num_provided < 5:
        raise TypeError("The following parameters must be provided: Om, h, w, s8, z.")

    # Print the input parameters
    if verbose:
        print(f"Received input parameters: Om={Om}, h={h}, w={w}, s8={s8}, z={z}")

    X_input = transform_input(*params)
    
    # Prepare input data in DMatrix format for XGBoost
    dmatrix = xgb.DMatrix(X_input)


    components_predictions = []
    # Load the pre-trained XGBoost model
    for i in range(4):  # Assuming we have 4 components/models
        model_name = f'model_comp{i}'  # Dynamically generate model name (e.g., model_comp0, model_comp1, etc.)
        model = load_model(model_name)  # Load the corresponding model using the load_model function

        # Perform prediction with the current model
        pred = model.predict(dmatrix)
        
        # Store the predicted component
        components_predictions.append(pred)

    components_predictions = np.array(components_predictions).reshape(1,-1)

    # Inverse PCA transformation
    pca = load_pca()
    pdfs_reconstructed = pca.inverse_transform(components_predictions)

    # Defining the mu vector according the training set
    mu_vec_std = np.linspace(-2, 8, 5000)

    pdfs_trained = enforce_monotonicity(pdfs_reconstructed[0])
    pdf_smoothed = smooth_pdf_fourier(pdfs_trained, cutoff=10, x=mu_vec_std)


    model_mean = load_model("model_mean")  
    model_sigma = load_model("model_sigma") 

    mean = model_mean.predict(dmatrix)
    sigma = model_sigma.predict(dmatrix)

    mu_vec = mu_vec_std * sigma + mean
    pdf_non_std = pdf_smoothed / sigma

    mu_vec_prime = mu_vec / mean
    pdf_prime = pdf_non_std * mean

    mu_center = (mu_vec_prime[1:] - mu_vec_prime[:-1]) / 2
    dx = np.diff(mu_vec_prime)
    pdf_prime_normalized = pdf_prime / np.sum(pdf_prime*dx)

    print("\nMean and Normalization Check:")
    print(f"Normalization: {np.sum(pdf_prime_normalized * dx)}")
    print(f"Mean: {np.sum(mu_center * pdf_prime_normalized * dx)}")

    return mu_vec_prime, pdf_prime_normalized


def predict_sigma(Om: float, h: float, w: float,  s8: float, z: float, verbose=True):
    """
    Predicts the sigma (standard deviation) for the PDF based on input cosmological parameters.

    Parameters:
    -----------
    Om : float, optional
        Matter density parameter.
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    z : float
        Redshift.
    verbose : bool, optional
        If True, prints the input parameters. Defaults to True.
        
    Returns:
    --------
    sigma : numpy.ndarray
        Predicted sigma value for the PDF.

    Raises:
    -------
    ValueError
        If any mandatory parameter (Om, h, w, s8, z) is missing.
    """

    # Count the number of parameters provided
    params = [Om, h, w, s8, z]
    num_provided = sum(p is not None for p in params)

    check_parameters(*params)

    if num_provided < 5:
        raise TypeError("The following parameters must be provided: Om, h, w, s8, z.")

    # Print the input parameters
    if verbose:
        print(f"Received input parameters: Om={Om}, h={h}, w={w}, s8={s8}, z={z}")

    X_input = transform_input(*params)

    dmatrix = xgb.DMatrix(X_input)
    model = load_model("model_sigma")   
    return model.predict(dmatrix)
    

def predict_mean(Om: float, h: float, w: float,  s8: float, z: float, verbose=True):
    """
    Predicts the mean for the PDF based on input cosmological parameters.

    Parameters:
    -----------
    Om : float, optional
        Matter density parameter.
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    z : float
        Redshift.
    verbose : bool, optional
        If True, prints the input parameters. Defaults to True.

    Returns:
    --------
    mean : numpy.ndarray
        Predicted mean value for the PDF.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """
    # Count the number of parameters provided
    params = [Om, h, w, s8, z]
    num_provided = sum(p is not None for p in params)

    check_parameters(*params)

    if num_provided < 5:
        raise TypeError("The following parameters must be provided: Om, h, w, s8, z.")

    # Print the input parameters
    if verbose:
        print(f"Received input parameters: Om={Om}, h={h}, w={w}, s8={s8}, z={z}")

    X_input = transform_input(*params)

    dmatrix = xgb.DMatrix(X_input)
    model = load_model("model_mean")   
    return model.predict(dmatrix)



def transform_input(Om: float, h: float, w: float,  s8: float, z: float):
    """
    Validates and transforms the input parameters into a standardized format for model prediction.

    Parameters:
    -----------
    Om : float, optional
        Matter density parameter.
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    z : float
        Redshift.

    Returns:
    --------
    X_input_std : numpy.ndarray
        The standardized input parameters for model prediction.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """

    X_input = np.array([Om, h, w, s8, z])

    # Define the column names
    column_names = ["Om", "h", "w", "s8", "z"]

    # Create the DataFrame
    df = pd.DataFrame(X_input.reshape(1, -1), columns=column_names)

    scale = load_scaling()
    X_input_scaled = scale.transform(df)

    return X_input_scaled
