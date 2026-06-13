"""
polyfit — command-line entry point.

Usage
-----
    python main.py              # uses input.ini
    python main.py my_run.ini   # uses a custom config file

The model is chosen automatically from the config:
    [data_structure]
    high_conversion = false   ->  Mayo-Lewis instantaneous model
    high_conversion = true    ->  Skeist integrated model (X > 5 %)

See README.md for the full configuration reference.
"""

import sys
import configparser

from source.fit_model import FitModel
from source.models import (mayo_lewis, mayo_lewis_deriv,
                            skeist, skeist_deriv, skeist_deriv_f0)


def _is_high_conversion(config_file: str) -> bool:
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    return cfg.getboolean('data_structure', 'high_conversion', fallback=False)


config_file = sys.argv[1] if len(sys.argv) > 1 else 'input.ini'

if _is_high_conversion(config_file):
    print("Model: Skeist integrated equation (high conversion)")
    fit = FitModel(skeist, skeist_deriv, config_file, derivative_model_f0=skeist_deriv_f0)
else:
    print("Model: Mayo-Lewis instantaneous equation")
    fit = FitModel(mayo_lewis, mayo_lewis_deriv, config_file)

if fit.run_pte:
    fit.run_pte_test(N_mock=fit.N_pte, output_file=fit.pte_output_file)

