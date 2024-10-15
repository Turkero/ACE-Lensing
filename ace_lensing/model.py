"""
model.py

This module contains functions for predicting lensing probability density functions (pdf) using a pre-trained XGBoost model.
It includes input validation to ensure that the necessary parameters are provided for making predictions.

Functions:
- predict(h, w, z, Om=None, s8=None, S8=None): Predicts pdf based on cosmological parameters.
"""

import xgboost as xgb
import numpy as np
import pkg_resources
import xgboost as xgb
import pandas as pd
import pickle
import logging

# Configure the logging system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name):
    """
    Load a pre-trained XGBoost model.

    Parameters:
    - model_name (str): Name of the model to load.

    Returns:
    - xgb.Booster: Loaded XGBoost model.
    """
    model_path = pkg_resources.resource_filename('ace_lens', f'models/{model_name}.xgb')
    logging.info(f"Loading model: {model_name} from {model_path}")
    try:
        model = xgb.Booster(model_file=model_path)
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading model: {e}")
        raise

def load_training_data():
    """
    Load the training data.

    Returns:
    - pd.DataFrame: Training data in pandas DataFrame format.
    """

    data_path = pkg_resources.resource_filename('ace_lens', 'data/training_data.csv')
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Training data file not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading training data: {e}")
        raise

def load_pca():
    """
    Load PCA (Principal Component Analysis) transformation model.

    Returns:
    - PCA: Loaded PCA model.
    """
    pca_path = pkg_resources.resource_filename('ace_lens', 'models/pca.pkl')
    try:
        with open(pca_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"PCA model file not found: {pca_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading PCA model: {e}")
        raise

    
def load_scaling():
    """
    Load input data scaling model.

    Returns:
    - scale: The scaling model.
    """
    scale_path = pkg_resources.resource_filename('ace_lens', 'models/scale.pkl')
    logging.info(f"Loading scaling model from {scale_path}")
    try:
        with open(scale_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"Scaling model file not found: {scale_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading scaling model: {e}")
        raise


def predict(h: float, w: float, z: float, Om: float = None, s8: float = None, S8: float = None):
    """
    Predicts the probability density function (PDF) based on input cosmological parameters 
    using pre-trained models and inverse PCA transformation.

    Parameters:
    -----------
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    z : float
        Redshift.
    Om : float, optional
        Matter density parameter. If not provided, it will be calculated using other parameters.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    S8 : float, optional
        A derived parameter related to s8 and Om. If not provided, it will be calculated using other parameters.

    Returns:
    --------
    mu_vec : numpy.ndarray
        The non-standardized vector of values (mu) for the PDF.
    pdf : numpy.ndarray
        The non-standardized reconstructed PDF.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """


    # Log the input parameters
    logging.info(f"Received input parameters: h={h}, w={w}, z={z}, Om={Om}, s8={s8}, S8={S8}")

    X_input = validate_and_transform_input(h, w, z, Om, s8, S8)

    components_predictions = []
    # Load the pre-trained XGBoost model
    for i in range(8):  # Assuming you have 8 components/models
        model_name = f'model_comp{i}'  # Dynamically generate model name (e.g., model_comp0, model_comp1, etc.)
        model = load_model(model_name)  # Load the corresponding model using the load_model function
        
        # Prepare input data in DMatrix format for XGBoost
        dmatrix = xgb.DMatrix(X_input)
        
        # Perform prediction with the current model
        pred = model.predict(dmatrix)
        
        # Store the predicted component
        components_predictions.append(pred)

    # Inverse PCA transformation
    pca = load_pca()
    pdfs_reconstructed = pca.inverse_transform(components_predictions)

    # Defining the mu vector according the training set
    mu_vec_std = np.linspace(-2, 8, 5000)
    # Normalization
    pdf_std = pdfs_reconstructed / np.trapz(pdfs_reconstructed, mu_vec_std)

    # Non-standardizing
    sigma = predict_sigma(X_input)
    mean = predict_mean(X_input)

    mu_vec = mu_vec_std * sigma + mean
    pdf_non_std = pdf_std / sigma
    pdf = pdf_non_std / np.trapz(pdf_non_std, mu_vec)

    return mu_vec, pdf


def predict_sigma(h: float, w: float, z: float, Om: float = None, s8: float = None, S8: float = None):
    """
    Predicts the sigma (standard deviation) for the PDF based on input cosmological parameters.

    Parameters:
    -----------
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    z : float
        Redshift.
    Om : float, optional
        Matter density parameter. If not provided, it will be calculated using other parameters.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    S8 : float, optional
        A derived parameter related to s8 and Om. If not provided, it will be calculated using other parameters.

    Returns:
    --------
    sigma : numpy.ndarray
        Predicted sigma value for the PDF.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """


    X_input = validate_and_transform_input(h, w, z, Om, s8, S8)

    dmatrix = xgb.DMatrix(X_input)
    model = load_model("model_sigma")   
    return model.predict(dmatrix)
    

def predict_mean(h: float, w: float, z: float, Om: float = None, s8: float = None, S8: float = None):
    """
    Predicts the mean for the PDF based on input cosmological parameters.

    Parameters:
    -----------
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    z : float
        Redshift.
    Om : float, optional
        Matter density parameter. If not provided, it will be calculated using other parameters.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    S8 : float, optional
        A derived parameter related to s8 and Om. If not provided, it will be calculated using other parameters.

    Returns:
    --------
    mean : numpy.ndarray
        Predicted mean value for the PDF.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """


    X_input = validate_and_transform_input(h, w, z, Om, s8, S8)

    dmatrix = xgb.DMatrix(X_input)
    model = load_model("model_mean")   
    return model.predict(dmatrix)



def validate_and_transform_input(h, w, z, Om, s8, S8):
    """
    Validates and transforms the input parameters into a standardized format for model prediction.

    Parameters:
    -----------
    h : float
        Hubble parameter.
    w : float
        Dark energy equation of state parameter.
    z : float
        Redshift.
    Om : float, optional
        Matter density parameter. If not provided, it will be calculated using other parameters.
    s8 : float, optional
        Standard deviation of matter fluctuations on a scale of 8 h^-1 Mpc. If not provided, it will be calculated using other parameters.
    S8 : float, optional
        A derived parameter related to s8 and Om. If not provided, it will be calculated using other parameters.

    Returns:
    --------
    X_input_std : numpy.ndarray
        The standardized input parameters for model prediction.

    Raises:
    -------
    ValueError
        If any mandatory parameter (h, w, z) is missing or fewer than two optional parameters (Om, s8, S8) are provided.
    """

    
    # Check that mandatory parameters are provided
    if h is None:
        logging.error("Parameter 'h' is mandatory.")
        raise ValueError("Parameter 'h' is mandatory.")
    if w is None:
        logging.error("Parameter 'w' is mandatory.")
        raise ValueError("Parameter 'w' is mandatory.")
    if z is None:
        logging.error("Parameter 'z' is mandatory.")
        raise ValueError("Parameter 'z' is mandatory.")

    # Validate that at least two out of the three optional parameters are provided
    provided = [Om, s8, S8]
    provided_count = sum(arg is not None for arg in provided)
    if provided_count < 2:
        logging.error("You must provide at least two of the following: Om, s8, S8.")
        raise ValueError("You must provide at least two of the following: Om, s8, S8.")
    
    # Correcting parameters
    S8 = s8 / np.sqrt( Om / 0.3) if S8 is None else S8
    Om = ( (s8 / S8)**2 ) * 0.3 if Om is None else Om
    s8 = S8 * np.sqrt( Om / 0.3) if s8 is None else s8
    X_input = np.array([Om, h, w, s8, S8, z])

    scale = load_scaling()
    X_input_std = scale.transform(X_input)

    return X_input_std