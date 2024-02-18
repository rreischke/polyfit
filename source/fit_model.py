import numpy as np
import matplotlib.pyplot as plt

from source.read_data import ReadData
from scipy.optimize import minimize
from scipy import optimize
from scipy.integrate import quad, dblquad, trapz, simps
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline, interp1d
from matplotlib.patches import Ellipse


class FitModel(ReadData):
    """
    Class providing structure to infer parameters from a non-linear model.
    Can calculate the best fit values for arbitrarily correlated independent
    and dependent variables assuming a Gaussian likelihood.

    Convidence intervals are calculated for the full distribution. Depending
    on the number of parameters, this can be done either by direct integration
    or by using MCMC methods.

    Parameters
    ----------
    model : function
        Python function which depends on the parameters. Describes the model
        as a function of the data.

    derivative_model : function
        Python function which depends on the parameters. Describes the
        derivative of the model as a function of the data.

    prior_range : array
        Numpy array with the prior range with shape (number of parameters, 2),
        first entry is the minimum and second entry the maximum.
    """

    def __init__(self,
                 model,
                 derivative_model,
                 config_name='input.ini'):
        super().__init__(config_name='input.ini')
        self.model = model
        self.derivative_model = derivative_model
        self.number_parameters = len(self.prior_range[:, 0])
        self.best_fit = None
        self.max_func = None
        self.parameter_x = np.linspace(
            self.prior_range[0, 0],
            self.prior_range[0, 1],
            self.N_inter)
        self.parameter_y = np.linspace(
            self.prior_range[1, 0],
            self.prior_range[1, 1],
            self.N_inter)
        self.evidence = 1.0
        self.find_MLE(np.ones(self.number_parameters))
        self.fisher_at_bf = self.fisher(self.best_fit)
        self.update_prior_range()
        self.normalize_posterior()
        self.quantile = None
        self.posterior_grid = None
        self.create_posterior_grid()
        self.p1_pdf_marginal_spline = None
        self.p2_pdf_marginal_spline = None
        self.p1_cdf_marginal_spline = None
        self.p2_cdf_marginal_spline = None
        self.best_fit_curve_plot()
        self.find_1d_marginal()
        self.create_contour_plot()
        self.print_give_result()

    def get_R(self, parameters):
        R = np.zeros((self.number_of_data_points, self.number_of_data_points))
        T = np.zeros((self.number_of_data_points, self.number_of_data_points))
        for i in range(self.number_of_data_points):
            T[i][i] = float(self.derivative_model(parameters, self.data_X[i]))
        R = self.CYY - np.dot(self.CXY, T)\
            - np.dot(T, self.CXY) + np.dot(T, np.dot(self.CXX, T))
        return R

    def partial_derivative_vec_arg(self, func, var=0, point=[]):
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(args)
        return derivative(wraps, point[var], dx=1e-5)

    def partial_derivative(self, func, data, var=0, point=[]):
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(args, data)
        return derivative(wraps, point[var], dx=1e-5)

    def chi_square(self, parameters):
        """
        Returns the chi2 for the model provided a covariance matrix

        Parameters:
        ----------
        parameters : array
            Parameters arrangend in an array.

        """
        for i in range(self.number_parameters):
            if(parameters[i] < self.prior_range[i, 0] or
               parameters[i] > self.prior_range[i, 1]):
                return 1e20
        Y_tilde = np.zeros([self.number_of_data_points])
        for i in range(self.number_of_data_points):
            Y_tilde[i] = self.data_Y[i] - \
                self.model(parameters, self.data_X[i])
        R = self.get_R(parameters)
        Rinverse = np.linalg.inv(R)
        return (0.5*(np.log(np.linalg.det(R))
                     + np.dot(Y_tilde, Rinverse.dot(Y_tilde))))

    def fisher(self, parameters):
        dR = np.zeros((2, self.number_of_data_points,
                       self.number_of_data_points))
        dmu = np.zeros((2, self.number_of_data_points))
        R = self.get_R(parameters)
        dR[0, :, :] = self.partial_derivative_vec_arg(
            self.get_R, 0, parameters)
        dR[1, :, :] = self.partial_derivative_vec_arg(
            self.get_R, 1, parameters)
        for i in range(self.number_of_data_points):
            dmu[0, i] = self.partial_derivative(
                self.model, self.data_X[i], 0, parameters)
            dmu[1, i] = self.partial_derivative(
                self.model, self.data_X[i], 1, parameters)
        Rinverse = np.linalg.inv(R)
        fisher_matrix = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                fisher_matrix[a][b] = 0.5*np.trace(
                    np.dot(np.dot(Rinverse, dR[a]),
                           np.dot(Rinverse, dR[b])) + np.dot(
                        Rinverse, np.outer(dmu[a],
                                           dmu[b]) + np.outer(dmu[b],
                                                              dmu[a])))
        return fisher_matrix

    def find_MLE(self, guess):
        """
        Function finding the maximum likelihood estimator

        Parameters:
        ----------
            guess: array
                Initial guess for the best fit parameters.
        """
        if(len(guess) != self.number_parameters):
            raise Exception(
                "Starting values does not have the right shape.")
        res = minimize(self.chi_square, guess, method='nelder-mead',
                       options={'xatol': 1e-6, 'disp': True})
        self.best_fit = np.array([res.x[0], res.x[1]])
        self.max_func = self.chi_square(res.x)

    def update_prior_range(self):
        invFisher = np.linalg.inv(self.fisher_at_bf)
        std_p1 = np.sqrt(invFisher[0, 0])
        std_p2 = np.sqrt(invFisher[1, 1])
        shift = [10.0*std_p1, 10.0*std_p2]
        for i in range(self.number_parameters):
            if(self.best_fit[i] - shift[i] > self.prior_range[i, 0]):
                self.prior_range[i, 0] = self.best_fit[i] - shift[i]
            if(self.best_fit[i] + shift[i] < self.prior_range[i, 1]):
                self.prior_range[i, 1] = self.best_fit[i] + shift[i]
        self.parameter_x = np.linspace(
            self.prior_range[0, 0],
            self.prior_range[0, 1],
            self.N_inter)
        self.parameter_y = np.linspace(
            self.prior_range[1, 0],
            self.prior_range[1, 1],
            self.N_inter)

    def posterior(self, parameters):
        return self.evidence*np.exp(-(self.chi_square(parameters)-self.max_func))

    def posterior_integrate(self, x, y):
        return self.evidence * np.exp(-(self.chi_square([x, y])-self.max_func))

    def create_posterior_grid(self):
        self.posterior_grid = np.zeros((self.N_inter, self.N_inter))
        print("creating posterior grid")
        print("...")
        for i in range(self.N_inter):
            for j in range(self.N_inter):
                self.posterior_grid[i, j] = self.posterior(
                    [self.parameter_x[i], self.parameter_y[j]])

    def find_contour_root(self, threshold):
        inner_integral = np.zeros(self.N_inter)
        for i in range(self.N_inter):
            fofx = np.zeros(self.N_inter)
            for j in range(self.N_inter):
                if(self.posterior_grid[i][j] > threshold):
                    fofx[j] = self.posterior_grid[i][j]
            inner_integral[i] = simps(fofx, self.parameter_y)
        result = simps(inner_integral, self.parameter_x)
        return (result - self.quantile)

    def find_contour(self, quantile):
        self.quantile = quantile
        max = np.amax(self.posterior_grid)
        min = np.amin(self.posterior_grid)
        sol = optimize.root_scalar(self.find_contour_root, bracket=[
            min, max], method='brentq', rtol=1e-4)
        return sol.root

    def normalize_posterior(self):
        result = dblquad(lambda x, y: self.posterior_integrate(
            x, y),
            self.prior_range[1, 0],
            self.prior_range[1, 1],
            lambda y: self.prior_range[0, 0],
            lambda y: self.prior_range[0, 1])
        self.evidence = 1.0 / result[0]

    def create_contour_plot(self):
        covariance = np.linalg.inv(self.fisher_at_bf)
        semi_a = np.sqrt((covariance[0, 0]+covariance[1, 1])/2.0+np.sqrt(
            (covariance[0, 0]-covariance[1, 1])**2.0/4.0 + covariance[0, 1]**2.0))
        semi_b = np.sqrt((covariance[0, 0]+covariance[1, 1])/2.0-np.sqrt(
            (covariance[0, 0]-covariance[1, 1])**2.0/4.0 + covariance[0, 1]**2.0))
        theta = np.arctan2(
            semi_a**2.0-covariance[0, 0], covariance[0, 1])/np.pi*180.0
        fontsi = 20
        plt.tick_params(labelsize=fontsi)
        if(self.use_tex):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '16'
        plt.rcParams['ytick.labelsize'] = '16'
        self.c1 = self.find_contour(0.683)
        self.c2 = self.find_contour(0.954)
        self.c3 = self.find_contour(0.997)
        fig, ax = plt.subplots()
        CS = ax.contour(self.parameter_x,
                        self.parameter_y,
                        self.posterior_grid.T,
                        levels=[self.c3, self.c2, self.c1],
                        colors=[self.plot_line_colour,
                                self.plot_line_colour,
                                self.plot_line_colour])
        fmt = {}
        strs = [r'$99.7\%$', r'$95.4\%$', r'$68.3\%$']
        for l, s in zip(CS.levels, strs):
            fmt[l] = s
        ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
        plt.xlabel(self.plot_parameter_1, fontsize=fontsi)
        plt.ylabel(self.plot_parameter_2, fontsize=fontsi)
        plt.plot(self.best_fit[0], self.best_fit[1],
                 marker="x", color=self.plot_data_colour)
        plt.xlim(self.contour_plot_zoom *
                 self.parameter_x[0], self.parameter_x[self.N_inter-1]/self.contour_plot_zoom)
        plt.ylim(self.contour_plot_zoom *
                 self.parameter_y[0], self.parameter_y[self.N_inter-1]/self.contour_plot_zoom)
        ellipse = Ellipse((self.best_fit[0], self.best_fit[1]), width=semi_a * 2 * 1.51, height=semi_b *
                          2*1.51, angle=theta, edgecolor='red', facecolor='none', ls='-', alpha=0.5)
        ax.add_patch(ellipse)
        ellipse = Ellipse((self.best_fit[0], self.best_fit[1]), width=semi_a * 2 * 2.48, height=semi_b *
                          2*2.48, angle=theta, edgecolor='red', facecolor='none', ls='-', alpha=0.5)
        ax.add_patch(ellipse)
        ellipse = Ellipse((self.best_fit[0], self.best_fit[1]), width=semi_a * 2 * 3.44, height=semi_b *
                          2*3.44, angle=theta, edgecolor='red', facecolor='none', ls='-', alpha=0.5)
        ax.add_patch(ellipse)
        plt.tight_layout()
        plt.savefig(self.file_name_contour_plot)

    def best_fit_curve_plot(self):
        fontsi = 20
        fontsi2 = 16
        plt.tick_params(labelsize=fontsi)
        if(self.use_tex):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '16'
        plt.rcParams['ytick.labelsize'] = '16'
        fig, ax1 = plt.subplots()
        plt.errorbar(self.data_X, self.data_Y, np.sqrt(np.diagonal(self.CYY)), np.sqrt(
            np.diagonal(self.CXX)), marker='.', ls='', label=r"data", color=self.plot_data_colour)
        self.Nfid = 10000
        self.Ybf = np.zeros([self.Nfid])
        self.Ybfp = np.zeros([self.Nfid])
        self.Ybfm = np.zeros([self.Nfid])
        fac = 4.0
        xmin = 0. # self.data_X[0]-fac*np.sqrt(self.CXX[0, 0])
        xmax = 1.0 # self.data_X[self.number_of_data_points-1] + \
            #fac*np.sqrt(self.CXX[self.number_of_data_points -
            #                     1, self.number_of_data_points-1])
        ymin =  0. # self.data_Y[0]-fac*np.sqrt(self.CYY[0, 0])
        ymax = 1. # self.data_Y[self.number_of_data_points-1] + \
            #fac*np.sqrt(self.CYY[self.number_of_data_points -
            #                     1, self.number_of_data_points-1])
        self.Xfid = np.linspace(0, 1.0, self.Nfid)
        for i in range(self.Nfid):
            self.Ybf[i] = self.model(
                [self.best_fit[0], self.best_fit[1]], self.Xfid[i])
        plt.plot(self.Xfid, self.Ybf, ls='-', label=r"best fit model",
                 color=self.plot_line_colour)
        plt.xlabel(self.plot_independent_variable, fontsize=fontsi)
        plt.ylabel(self.plot_dependent_variable, fontsize=fontsi)
        plt.legend(fancybox=True, loc='upper left', fontsize=fontsi2)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(self.file_name_best_fit_plot)

    def get_1d_gaussian(self, x, mu, sigma):
        return 1.0/np.sqrt(2.0*np.pi*sigma**2.0)*np.exp(-0.5*(x-mu)**2.0/sigma**2.0)

    def find_1d_marginal(self):
        invFisher = np.linalg.inv(self.fisher_at_bf)
        std_p1 = np.sqrt(invFisher[0, 0])
        std_p2 = np.sqrt(invFisher[1, 1])
        fofx = np.zeros(self.N_inter)
        p1_marginal = simps(self.posterior_grid, self.parameter_y, axis= -1)
        p2_marginal = simps(self.posterior_grid, self.parameter_x, axis= -2)
        self.p1_pdf_marginal_spline = interp1d(
            self.parameter_x, p1_marginal, fill_value="extrapolate")
        self.p2_pdf_marginal_spline = interp1d(
            self.parameter_y, p2_marginal, fill_value="extrapolate")
        p1_cdf_marginal = np.zeros(self.N_inter)
        p2_cdf_marginal = np.zeros(self.N_inter)
        for i in range(self.N_inter-1):
            aux_int_x = np.linspace(
                self.parameter_x[0], self.parameter_x[i+1], 1000)
            aux_int_y = np.linspace(
                self.parameter_y[0], self.parameter_y[i+1], 1000)
            p1_cdf_marginal[i] = trapz(
                self.p1_pdf_marginal_spline(aux_int_x), aux_int_x)
            p2_cdf_marginal[i] = trapz(
                self.p2_pdf_marginal_spline(aux_int_y), aux_int_y)
        p1_cdf_marginal[self.N_inter-1] = 1.0
        p2_cdf_marginal[self.N_inter-1] = 1.0
        self.p1_cdf_marginal_spline = interp1d(
            p1_cdf_marginal, self.parameter_x, fill_value="extrapolate")
        self.p2_cdf_marginal_spline = interp1d(
            p2_cdf_marginal, self.parameter_y, fill_value="extrapolate")

        fontsi = 20
        fontsi2 = 16
        plt.tick_params(labelsize=fontsi)
        if(self.use_tex):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
        plt.rc('font', family='Arial')
        plt.rcParams['xtick.labelsize'] = '16'
        plt.rcParams['ytick.labelsize'] = '16'
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(self.plot_parameter_1, fontsize=fontsi)
        ax1.set_ylabel(r"marginal posterior distribution", fontsize=fontsi)
        ax1.plot(self.parameter_x, self.p1_pdf_marginal_spline(
            self.parameter_x), color=self.plot_line_colour, label=r"exact")
        ax1.plot(self.parameter_x, self.get_1d_gaussian(
            self.parameter_x, self.best_fit[0], std_p1), ls="-", color="red", label=r"Gaussian")
        ax1.set_xlim(self.contour_plot_zoom *
                     self.parameter_x[0], self.parameter_x[self.N_inter-1]/self.contour_plot_zoom)
        ax1.legend(fancybox=True, loc='upper right', fontsize=fontsi2)
        plt.tight_layout()
        plt.savefig(self.file_name_1d_marginal_p1_plot)

        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(self.plot_parameter_2, fontsize=fontsi)
        ax2.set_ylabel(r"marginal posterior distribution", fontsize=fontsi)
        ax2.plot(self.parameter_y, self.p2_pdf_marginal_spline(
            self.parameter_y), color=self.plot_line_colour, label=r"exact")
        ax2.plot(self.parameter_y, self.get_1d_gaussian(
            self.parameter_y, self.best_fit[1], std_p2), ls="-", color="red", label=r"Gaussian")
        ax2.set_xlim(self.contour_plot_zoom *
                     self.parameter_y[0], self.parameter_y[self.N_inter-1]/self.contour_plot_zoom)
        ax2.legend(fancybox=True, loc='upper right', fontsize=fontsi2)

        plt.tight_layout()
        plt.savefig(self.file_name_1d_marginal_p2_plot)

    def print_give_result(self):
        invFisher = np.linalg.inv(self.fisher_at_bf)
        std_p1 = np.sqrt(invFisher[0, 0])
        std_p2 = np.sqrt(invFisher[1, 1])
        with open(self.file_name_results, 'w') as paramfile:
            paramfile.write("### Results\n\n")
            paramfile.write("The best fit values are: (p1,p2) = (" +
                            str(self.best_fit[0]) + ", " + str(self.best_fit[1]) + ")\n\n")
            paramfile.write("Corresponding symmetric one sigma Gaussian errors are: (" +
                            str(std_p1) + "," + str(std_p2) + ")\n\n")
            paramfile.write("Corresponding real left errors are: -(" +
                            str(self.best_fit[0] - self.p1_cdf_marginal_spline(0.16)) +
                            "," + str(self.best_fit[1] - self.p2_cdf_marginal_spline(0.16)) + ")\n\n")
            paramfile.write("Corresponding real right errors are: +(" +
                            str(self.p1_cdf_marginal_spline(0.84)-self.best_fit[0]) +
                            "," + str(self.p2_cdf_marginal_spline(0.84)-self.best_fit[1]) + ")\n\n")
            paramfile.write("The contour levels to plot the confidence intervals from the posterior text files are: "
                            + str(self.c1) + ", " + str(self.c2) + ", " + str(self.c3) + " for 68.3%, 95.4%, and 99.7%, respectively \n\n")

        with open(self.file_name_best_fit, 'w') as paramfile:
            paramfile.write("### Best fit curve\n")
            paramfile.write("### independent variable   dependent variable\n")
            for i in range(self.Nfid):
                paramfile.write(
                    str(self.Xfid[i]) + " " + str(self.Ybf[i]) + "\n")
        with open(self.file_name_contour, 'w') as paramfile:
            paramfile.write("### 2D posterior distribution\n")
            paramfile.write("### parameter_1    parameter_2    posterior \n")
            for i in range(self.N_inter):
                for j in range(self.N_inter):
                    paramfile.write(
                        str(self.parameter_x[i]) + " " +
                        str(self.parameter_y[j]) + " " +
                        str(self.posterior_grid[i, j]) + "\n")
