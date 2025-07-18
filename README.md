# ACE Lens

**Accurate Cosmological Emulator for the Lensing PDF**

This project aims to provide a powerful cosmological emulator that predicts the lensing PDF with high accuracy.

## Installation

Clone the repository:
```bash
git clone git@github.com:Turkero/ACE-Lensing.git

cd ACE-Lensing
```
Here we recommend to create a virtual environment:
```bash
conda create -n ACE-env python=3.9  
```

```bash
conda activate ACE-env
pip install . 
```

## Installation Instructions For Mac users

Before installing the package, ensure you have the required system dependencies:

- Run the following command to install `libomp`:

```bash
brew install libomp
```

args = {'h': 1, 'w': 2, 'z': 3, 'Om': 4}
predict_pdf(**args)

params = [1, 2, 3, 4]  # order must be known
predict_pdf(*params)
