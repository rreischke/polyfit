import numpy as np
import configparser


class ReadData():
    def __init__(self,
                 config_name='input.config'):
        self.data_X = None
        self.data_Y = None
        self.CXX = None
        self.CYY = None
        self.CXY = None
        self.file_data = None
        self.X_covariance = None
        self.Y_covariance = None
        self.XY_covariance = None
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
        self.read_config(config_name)
        self.read_data(self.file_data, self.X_covariance,
                       self.Y_covariance, self.XY_covariance)

    def read_config(self, config_name):
        config = configparser.ConfigParser()
        config.read(config_name)
        if 'data_structure' in config:
            if 'file_data' in config['data_structure']:
                self.file_data = config['data_structure']['file_data']
            if 'X_covariance' in config['data_structure']:
                self.X_covariance = config['data_structure']['X_covariance'].split(
                    ' ')
            if 'Y_covariance' in config['data_structure']:
                self.Y_covariance = config['data_structure']['Y_covariance'].split(
                    ' ')
            if 'XY_covariance' in config['data_structure']:
                self.XY_covariance = config['data_structure']['XY_covariance'].split(
                    ' ')

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

    def read_data(self,
                  file_data,
                  file_X_covariance,
                  file_Y_covariance,
                  file_XY_covariance):
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
                    for i in range(self.number_of_data_points):
                        for j in range(self.number_of_data_points):
                            self.CXX[i, j] = data_CXX[i, j]
                else:
                    for i in range(self.number_of_data_points):
                        self.CXX[i, i] = data_CXX[i, 0]**2.0
        else:
            if(file_X_covariance[0] == 'rel'):
                for i in range(self.number_of_data_points):
                    self.CXX[i, i] = (
                        float(file_X_covariance[1])*self.data_X[i])**2.0
            if(file_X_covariance[0] == 'abs'):
                for i in range(self.number_of_data_points):
                    self.CXX[i, i] = (float(file_X_covariance[1]))**2.0

        self.CYY = np.zeros(
            (self.number_of_data_points, self.number_of_data_points))
        if(file_Y_covariance[0] != 'rel' and file_Y_covariance[0] != 'abs'):
            if(file_Y_covariance != ""):
                data_CYY = np.loadtxt(file_Y_covariance[0])
                if(len(data_CYY[0, :]) == self.number_of_data_points):
                    for i in range(self.number_of_data_points):
                        for j in range(self.number_of_data_points):
                            self.CYY[i, j] = data_CYY[i, j]
                else:
                    for i in range(self.number_of_data_points):
                        self.CYY[i, i] = data_CYY[i, 0]**2.0
            else:
                raise Exception("No covariance for the dependent variable"
                                + "defined, fitting will be ill-behaved. "
                                + "Check the path for the covariance.")
        else:
            if(file_Y_covariance[0] == 'rel'):
                for i in range(self.number_of_data_points):
                    self.CYY[i, i] = (
                        float(file_Y_covariance[1])*self.data_Y[i])**2.0
            if(file_Y_covariance[0] == 'abs'):
                for i in range(self.number_of_data_points):
                    self.CYY[i, i] = (float(file_Y_covariance[1]))**2.0

        self.CXY = np.zeros(
            (self.number_of_data_points, self.number_of_data_points))
        if(file_XY_covariance[0] != 'rel' and file_XY_covariance[0] != 'abs'):
            if(file_XY_covariance != ""):
                data_CXY = np.loadtxt(file_XY_covariance[0])
                if(len(data_CXY[0, :]) == self.number_of_data_points):
                    for i in range(self.number_of_data_points):
                        for j in range(self.number_of_data_points):
                            self.CXY[i, j] = data_CXY[i, j]
                else:
                    for i in range(self.number_of_data_points):
                        self.CXY[i, i] = data_CXY[i, 0]**2.0
        else:
            if(file_XY_covariance[0] == 'rel'):
                for i in range(self.number_of_data_points):
                    self.CXY[i, i] = (float(file_XY_covariance[1]))**2.0 * \
                        self.data_X[i]*self.data_Y[i]
            if(file_XY_covariance[0] == 'abs'):
                for i in range(self.number_of_data_points):
                    self.CXY[i, i] = (float(file_XY_covariance[1]))**2.0
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
