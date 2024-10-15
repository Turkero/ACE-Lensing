from setuptools import setup, find_packages

setup(
    name='ace_lens',                    # Name of the package
    version='0.1.0',                    # Package version
    author='tunc',                      # Package author
    author_email='ozgen.turker@edu.ufes.br',  # Author's email
    description='Accurate Cosmological Emulator for the Lensing PDF',  # Short description
    long_description=open('README.md').read(),      # Read long description from README
    long_description_content_type='text/markdown',  # Format of long description
    url='https://github.com/Turkero/ace_lensing',     # URL to the package repository
    packages=find_packages(),           # Packages in the current directory
    include_package_data=True,          # Include data files as per MANIFEST.in
    classifiers=[                       # Classifiers for the package ############## I HAVE NO IDEA WHATS THAT
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',            # Python version requirement
    install_requires=[                  # Dependencies
    'wheel',                            # For installing packages in a binary format. pip can create .whl files, easier dependency resolution.
    'numpy>=2.0.2',                     # For numerical operations
    'pandas>=2.2.3',                    # For data manipulation and analysis
    'matplotlib>=3.9.2',                # For plotting
    'natsort>=8.3.1',                   # For natural sorting
    'scipy>=1.13.1',                    # For scientific computations
    'scikit-learn>=1.5.2',              # For machine learning tasks
    'xgboost>=2.0.1',                   # For gradient boosting
    ],
)

