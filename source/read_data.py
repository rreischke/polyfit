import os
import numpy as np
import configparser


class ReadData():
    def __init__(self,
                 config_name='input.config'):
        self.data_X = None
        self.data_Y = None
        self.data_f0 = None        # f0 column for high-conversion (Skeist) data
        self.high_conversion = False
        self.CXX = None
        self.CYY = None
        self.CXY = None
        self.CFF = None            # covariance on f0 (Skeist only); zeros if unused
        self.file_data = None
        self.X_covariance = None
        self.Y_covariance = None
        self.XY_covariance = None
        self.f0_covariance = None  # config-based f0 uncertainty (plain-text path)
        self.number_of_data_points = None
        self.N_inter = None
        self.prior_range = np.zeros((2, 2))
        self.plot_line_colour = None
        self.plot_data_colour = None
        self.plot_independent_variable = None
        self.plot_dependent_variable = None
        self.plot_parameter_1 = None
        self.plot_parameter_2 = None
        self.file_name_contour_plot = None
        self.file_name_best_fit_plot = None
        self.file_name_1d_marginal_p1_plot = None
        self.file_name_1d_marginal_p2_plot = None
        self.contour_plot_zoom = None
        self.file_name_best_fit = None
        self.file_name_contour = None
        self.file_name_results = None
        self.use_tex = None
        self.run_pte = False
        self.N_pte = 500
        self.pte_output_file = 'pte_test.pdf'
        self.read_config(config_name)
        self.read_data(self.file_data, self.X_covariance,
                       self.Y_covariance, self.XY_covariance)

    def read_config(self, config_name):
        config = configparser.ConfigParser()
        config.read(config_name)
        if 'data_structure' in config:
            if 'file_data' in config['data_structure']:
                self.file_data = config['data_structure']['file_data']
            if 'high_conversion' in config['data_structure']:
                self.high_conversion = config['data_structure'].getboolean(
                    'high_conversion', fallback=False)
            if 'X_covariance' in config['data_structure']:
                self.X_covariance = config['data_structure']['X_covariance'].split(
                    ' ')
            if 'Y_covariance' in config['data_structure']:
                self.Y_covariance = config['data_structure']['Y_covariance'].split(
                    ' ')
            if 'XY_covariance' in config['data_structure']:
                self.XY_covariance = config['data_structure']['XY_covariance'].split(
                    ' ')
            if 'f0_covariance' in config['data_structure']:
                self.f0_covariance = config['data_structure']['f0_covariance'].split(' ')

        if 'plotting' in config:
            if 'plot_line_colour' in config['plotting']:
                self.plot_line_colour = config['plotting']['plot_line_colour']
            if 'plot_data_colour' in config['plotting']:
                self.plot_data_colour = config['plotting']['plot_data_colour']
            if 'independent_variable_name' in config['plotting']:
                self.plot_independent_variable = config['plotting']['independent_variable_name']
                self.plot_independent_variable = r'{}'.format(
                    self.plot_independent_variable)
            if 'dependent_variable_name' in config['plotting']:
                self.plot_dependent_variable = config['plotting']['dependent_variable_name']
                self.plot_dependent_variable = r'{}'.format(
                    self.plot_dependent_variable)
            if 'parameter_1_name' in config['plotting']:
                self.plot_parameter_1 = config['plotting']['parameter_1_name']
                self.plot_parameter_1 = r'{}'.format(self.plot_parameter_1)
            if 'parameter_2_name' in config['plotting']:
                self.plot_parameter_2 = config['plotting']['parameter_2_name']
                self.plot_parameter_2 = r'{}'.format(self.plot_parameter_2)
            if 'contour_plot_zoom' in config['plotting']:
                self.contour_plot_zoom = float(
                    config['plotting']['contour_plot_zoom'])
            if 'use_tex' in config['plotting']:
                self.use_tex = config['plotting'].getboolean('use_tex')

        if 'precision' in config:
            if 'N_inter' in config['precision']:
                self.N_inter = int(config['precision']['N_inter'])

        if 'inference' in config:
            if 'prior_range' in config['inference']:
                aux = config['inference']['prior_range'].split(', ')
                self.prior_range[0, 0] = float(aux[0])
                self.prior_range[0, 1] = float(aux[1])
                self.prior_range[1, 0] = float(aux[2])
                self.prior_range[1, 1] = float(aux[3])
            if 'run_pte' in config['inference']:
                self.run_pte = config['inference'].getboolean('run_pte', fallback=False)
            if 'N_pte' in config['inference']:
                self.N_pte = int(config['inference']['N_pte'])

        if 'output' in config:
            if 'file_name_contour_plot' in config['output']:
                self.file_name_contour_plot = config['output']['file_name_contour_plot']
            if 'file_name_best_fit_plot' in config['output']:
                self.file_name_best_fit_plot = config['output']['file_name_best_fit_plot']
            if 'file_name_contour' in config['output']:
                self.file_name_contour = config['output']['file_name_contour']
            if 'file_name_best_fit' in config['output']:
                self.file_name_best_fit = config['output']['file_name_best_fit']
            if 'file_name_results' in config['output']:
                self.file_name_results = config['output']['file_name_results']
            if 'file_name_1d_marginal_p1_plot' in config['output']:
                self.file_name_1d_marginal_p1_plot = config['output']['file_name_1d_marginal_p1_plot']
            if 'file_name_1d_marginal_p2_plot' in config['output']:
                self.file_name_1d_marginal_p2_plot = config['output']['file_name_1d_marginal_p2_plot']
            if 'pte_output_file' in config['output']:
                self.pte_output_file = config['output']['pte_output_file']

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_csv(filename):
        """Return True if the file appears to be comma-delimited."""
        if os.path.splitext(filename)[1].lower() == '.csv':
            return True
        with open(filename, 'r') as fh:
            first = fh.readline()
        return ',' in first

    @staticmethod
    def _load_csv(filename):
        """
        Load a comma-separated file, skipping a header row when present.

        A row is treated as a header if its first field cannot be parsed as a
        floating-point number.
        """
        with open(filename, 'r') as fh:
            first = fh.readline().strip()
        try:
            float(first.split(',')[0].strip())
            skiprows = 0
        except ValueError:
            skiprows = 1
        return np.loadtxt(filename, delimiter=',', skiprows=skiprows)

    # ------------------------------------------------------------------
    # Main data loader
    # ------------------------------------------------------------------

    def read_data(self,
                  file_data,
                  file_X_covariance,
                  file_Y_covariance,
                  file_XY_covariance):
        """
        Load data and covariance matrices.

        Supported file formats
        ----------------------
        Plain text  (2 columns, whitespace-separated, no header)
            col 1 : X   — independent variable (e.g. feed composition f2)
            col 2 : F   — dependent variable   (e.g. copolymer composition F2)
            Uncertainties are taken from the config (X_covariance, Y_covariance,
            XY_covariance).

        CSV standard  (4 columns, comma-separated, optional header row)
            col 1 : f0  — nominal / initial feed composition (metadata, not used in fit)
            col 2 : X   — measured independent variable
            col 3 : F   — measured dependent variable
            col 4 : dF  — absolute uncertainty on F (1 sigma)
            Uncertainties on X and f0 are assumed zero; XY cross-covariance is zero.

        CSV EVM  (5 columns, comma-separated, optional header row)
            col 1 : f0  — nominal / initial feed composition
            col 2 : X   — measured independent variable
            col 3 : dX  — absolute uncertainty on X (1 sigma)
            col 4 : F   — measured dependent variable
            col 5 : dF  — absolute uncertainty on F (1 sigma)
            EVM = Errors-in-Variables Model.  f0 uncertainty assumed zero.

        CSV EVM + f0 error  (6 columns, comma-separated, optional header row)
            col 1 : f0  — nominal / initial feed composition
            col 2 : X   — measured independent variable
            col 3 : dX  — absolute uncertainty on X (1 sigma)
            col 4 : F   — measured dependent variable
            col 5 : dF  — absolute uncertainty on F (1 sigma)
            col 6 : df0 — absolute uncertainty on f0 (1 sigma)
            Used with the Skeist model (high_conversion = true) when the
            prepared feed composition itself carries measurement uncertainty.
            Requires FitModel to be constructed with derivative_model_f0.

        For CSV formats the covariance settings in the config file (X_covariance,
        Y_covariance, XY_covariance) are ignored — the uncertainties come
        directly from the delta columns in the file.  f0_covariance from the
        config is applied to plain-text files only.
        """

        # ── CSV path ──────────────────────────────────────────────────────
        if self._is_csv(file_data):
            data  = self._load_csv(file_data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            n_cols = data.shape[1]
            N = data.shape[0]
            self.number_of_data_points = N

            self.CXX = np.zeros((N, N))
            self.CYY = np.zeros((N, N))
            self.CXY = np.zeros((N, N))
            self.CFF = np.zeros((N, N))

            if n_cols == 4:
                # Standard CSV: f0, X, F, dF  (no X or f0 uncertainty)
                self.data_X = data[:, 1]
                self.data_Y = data[:, 2]
                np.fill_diagonal(self.CYY, data[:, 3] ** 2)
                if self.high_conversion:
                    self.data_f0 = data[:, 0]

            elif n_cols == 5:
                # EVM CSV: f0, X, dX, F, dF  (X uncertainty; no f0 uncertainty)
                self.data_X = data[:, 1]
                self.data_Y = data[:, 3]
                np.fill_diagonal(self.CXX, data[:, 2] ** 2)
                np.fill_diagonal(self.CYY, data[:, 4] ** 2)
                if self.high_conversion:
                    self.data_f0 = data[:, 0]

            elif n_cols == 6:
                # EVM + f0 error CSV: f0, X, dX, F, dF, df0
                self.data_X = data[:, 1]
                self.data_Y = data[:, 3]
                np.fill_diagonal(self.CXX, data[:, 2] ** 2)
                np.fill_diagonal(self.CYY, data[:, 4] ** 2)
                np.fill_diagonal(self.CFF, data[:, 5] ** 2)
                if self.high_conversion:
                    self.data_f0 = data[:, 0]

            else:
                raise ValueError(
                    f"CSV file '{file_data}' has {n_cols} columns.\n"
                    f"Expected 4 (f0, X, F, dF), 5 (f0, X, dX, F, dF), "
                    f"or 6 (f0, X, dX, F, dF, df0).\n"
                    f"See the README for the required CSV format."
                )
            return

        # ── Plain two-column text path ────────────────────────────────────
        data = np.loadtxt(file_data)
        self.number_of_data_points = len(data[:, 0])
        self.data_X = data[:, 0]
        self.data_Y = data[:, 1]

        self.CXX = np.zeros(
            (self.number_of_data_points, self.number_of_data_points))
        if(file_X_covariance[0] != 'rel' and file_X_covariance[0] != 'abs'):
            if(file_X_covariance != ""):
                data_CXX = np.loadtxt(file_X_covariance[0])
                if(len(data_CXX[0, :]) == self.number_of_data_points):
                    self.CXX[:] = data_CXX
                else:
                    np.fill_diagonal(self.CXX, data_CXX[:, 0]**2.0)
        else:
            if(file_X_covariance[0] == 'rel'):
                np.fill_diagonal(self.CXX, (float(file_X_covariance[1]) * self.data_X)**2.0)
            if(file_X_covariance[0] == 'abs'):
                np.fill_diagonal(self.CXX, float(file_X_covariance[1])**2.0)

        self.CYY = np.zeros(
            (self.number_of_data_points, self.number_of_data_points))
        if(file_Y_covariance[0] != 'rel' and file_Y_covariance[0] != 'abs'):
            if(file_Y_covariance != ""):
                data_CYY = np.loadtxt(file_Y_covariance[0])
                if(len(data_CYY[0, :]) == self.number_of_data_points):
                    self.CYY[:] = data_CYY
                else:
                    np.fill_diagonal(self.CYY, data_CYY[:, 0]**2.0)
            else:
                raise Exception("No covariance for the dependent variable"
                                + "defined, fitting will be ill-behaved. "
                                + "Check the path for the covariance.")
        else:
            if(file_Y_covariance[0] == 'rel'):
                np.fill_diagonal(self.CYY, (float(file_Y_covariance[1]) * self.data_Y)**2.0)
            if(file_Y_covariance[0] == 'abs'):
                np.fill_diagonal(self.CYY, float(file_Y_covariance[1])**2.0)

        self.CXY = np.zeros(
            (self.number_of_data_points, self.number_of_data_points))
        if(file_XY_covariance[0] != 'rel' and file_XY_covariance[0] != 'abs'):
            if(file_XY_covariance != ""):
                data_CXY = np.loadtxt(file_XY_covariance[0])
                if(len(data_CXY[0, :]) == self.number_of_data_points):
                    self.CXY[:] = data_CXY
                else:
                    np.fill_diagonal(self.CXY, data_CXY[:, 0]**2.0)
        else:
            if(file_XY_covariance[0] == 'rel'):
                np.fill_diagonal(self.CXY, float(file_XY_covariance[1])**2.0 * self.data_X * self.data_Y)
            if(file_XY_covariance[0] == 'abs'):
                np.fill_diagonal(self.CXY, float(file_XY_covariance[1])**2.0)
        if(file_XY_covariance != "" and file_X_covariance != ""):
            if(self.CXX.shape[0] != self.CYY.shape[0] or
               self.CXX.shape[0] != self.number_of_data_points or
               self.CYY.shape[0] != self.number_of_data_points):
                raise Exception(
                    "Files with covariance matrices do not match the size of\
                    the data")
        else:
            if(self.number_of_data_points != self.CYY.shape[0]):
                raise Exception(
                    "Files with covariance matrices do not match the size of\
                    the data")

        N = self.number_of_data_points
        self.CFF = np.zeros((N, N))
        if self.f0_covariance is not None:
            if self.f0_covariance[0] == 'rel':
                np.fill_diagonal(self.CFF,
                                 (float(self.f0_covariance[1]) * self.data_f0) ** 2.0
                                 if self.data_f0 is not None else 0.0)
            elif self.f0_covariance[0] == 'abs':
                np.fill_diagonal(self.CFF, float(self.f0_covariance[1]) ** 2.0)
            else:
                data_CFF = np.loadtxt(self.f0_covariance[0])
                if data_CFF.ndim == 2 and data_CFF.shape[0] == N:
                    self.CFF[:] = data_CFF
                else:
                    np.fill_diagonal(self.CFF, data_CFF.ravel()[:N] ** 2.0)
