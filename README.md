This package allows to infer copolymerization reactivity ratios for arbitrarily correlated data assuming a Gaussian covariance. The final inference takes the non-linearity of the model into account. Details on the methodology can be found in [Reischke 2023](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mats.202200063).

## Installation and Examples
For starters you first clone or download the directory
```shell
git clone git@github.com:rreischke/polyfit.git
```
or
```shell
git clone https://github.com/rreischke/polyfit.git
```
Then navigate to the cloned directory
```shell
cd polyfit
pip install .
```

Once you have installed the external package via ``pip install`` the code simply runs by using the ``input.ini`` where all parameters are stored and explained. Running the script
```shell
python main.py
```
will run the code using the settings in the standard configuration file ``input.ini``. You can always use your own ``my_input.ini`` and run via:
```shell
python main.py my_input.ini
```

If you simply run the standard ``input.ini`` it will use the data described in [Scott and Penlidis (2017)](https://www.mdpi.com/2227-9717/6/1/8) with a relative error of five percent on both the dependent and independent variable. The produced output will contain the three plots for the parameter inference, i.e. the 2 one dimensional marginals and the contour plot. The fourth plot is the data with the best fit curve. The data to produce these four plots is also stored in two ``.txt`` files named ``posterior_scott`` and ``bestfist_model_scott``. The results of the fitting are stored in ``results_scott.txt``, listing the pest fit value, the symmetric Gaussian error (corresponding to the red curves) and the real error on the parameters taking the non-linearity of the model into account (blue curve in the plot).

