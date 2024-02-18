This package allows to infer copolymerization reactivity ratios for arbitrarily correlated data assuming a Gaussian covariance. The final inference takes the non-linearity of the model into account. Details on the methodology can be found in [Reischke 2023](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mats.202200063)

## Installation and Examples
For starters you first clone the directory via:
```shell
git clone git@github.com:rreischke/OneCovariance.git
```
or
```shell
git clone https://github.com/rreischke/OneCovariance.git
```
Then navigate to the cloned directory
```shell
cd OneCovariance
conda env create -f conda_env.yaml
conda activate cov20_env
pip install .
```
On some Linux servers you will have to install ``gxx_linux-64`` by hand and the installation will not work. This usually shows the following error message in the terminal:
``
gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
``
If this is the case just install it by typing
```shell
 conda install -c conda-forge gxx_linux-64
```
and redo the ``pip`` installation.

If you do not want to use the conda environment make sure that you have ``gfortran`` and ``gsl`` installed.
You can install both via ``conda``:
```shell
conda install -c conda-forge gfortran
conda install -c conda-forge gsl
conda install -c conda-forge gxx_linux-64
git clone git@github.com:rreischke/OneCovariance.git
cd OneCovariance    
pip install .
```
Once you have installed the external package via ``pip install`` the code simply runs by using the ``config.ini`` where all parameters are stored and explained. Running the script
```shell
python covariance.py
```
will run the code using the settings in the standard configuration file ``config.ini``. 

