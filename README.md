# ACE-Lensing
### *Accurate Cosmological Emulator for the Lensing Magnification PDF*

`ace_lensing` provides a fast and accurate emulator for the probability distribution function (PDF) of gravitational lensing magnification for point sources.  
It includes the trained PCA decomposition and XGBoost regression models to load the standardized/non-standardized training datasets.

This package allows users to:
- predict a full magnification PDF for any cosmology–redshift point   
- predict the standard deviation of the magnification PDF alone
- load the training/testing datasets  
- inspect the original simulation data used during emulator construction  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Turkero/ACE-Lensing.git
cd ACE-Lensing
```

Create a virtual Python environment (optional but recommended):

```bash
conda create -n ACE-env python=3.9
conda activate ACE-env
```

Install the package:

```bash
pip install . 
```

### For Mac users

XGBoost requires OpenMP support. Install:
```bash
brew install libomp
```

## Running ACE-Lensing
### Predict the full magnification PDF

```python
from ace_lensing import predict_pdf

args = {"Om": 0.3, "h": 0.68, "w": -1.0, "s8": 0.81, "z": 1.0}
pdf, mu = predict_pdf(**args)
```

Using positional arguments:

```python
params = [0.3, 0.68, -1.0, 0.81, 1.0]
pdf, mu = predict_pdf(*params)
```

### Predict the Standard Deviation only
If you only need the standard deviation of the magnification PDF instead of the full PDF:

```python
from ace_lensing import predict_sigma

args = {"Om": 0.3, "h": 0.68, "w": -1.0, "s8": 0.81, "z": 1.0}
sigma = predict_sigma(**args)
print(sigma)
```

Or using positional arguments:
```python
params = [0.3, 0.68, -1.0, 0.81, 1.0]
sigma = predict_sigma(*params)
```

- sigma → predicted standard deviation of the magnification distribution

This is the same $\sigma$ used to de-standardize the PCA-reconstructed PDFs during training.


### Loading Training & Testing Data

```python
from ace_lensing.model import load_training_data, load_testing_data

train = load_training_data()
test = load_testing_data()
```

All data files are stored inside:
````
ace_lensing/data/
````

### Package Structure
```
|-- MANIFEST.in
|-- README.md
|-- ace_lensing
|   |-- __init__.py
|   |-- data
|   |   |-- testing_set.csv
|   |   |-- testing_set_cosmo.parquet
|   |   |-- testing_set_error.parquet
|   |   |-- testing_set_mu_vec.parquet
|   |   |-- testing_set_pdf.parquet
|   |   |-- testing_set_statistics.parquet
|   |   |-- training_set.csv
|   |   |-- training_set_cosmo.parquet
|   |   |-- training_set_error.parquet
|   |   |-- training_set_mu_vec.parquet
|   |   |-- training_set_pdf.parquet
|   |   `-- training_set_statistics.parquet
|   |-- model.py
|   `-- models
|       |-- model_comp0.xgb
|       |-- model_comp1.xgb
|       |-- model_comp2.xgb
|       |-- model_comp3.xgb
|       |-- model_mean.xgb
|       |-- model_sigma.xgb
|       |-- pca.pkl
|       `-- scale.pkl
|-- pyproject.toml
|-- setup.py
`-- tests
    `-- test.ipynb
```

---

### Citing

If you use this package in research:

<!-- Türker, Ö. T., et al. (2025).
Accurate Cosmological Emulator for the Probability Distribution Function of Gravitational Lensing of Point Sources. -->

---

### License

--- 

### Contact

For questions or suggestions, open a GitHub issue or contact:

📧 tuncturker.work@gmail.com