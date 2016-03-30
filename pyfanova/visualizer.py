import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from mpl_toolkits.mplot3d import Axes3D


class Visualizer(object):

    def __init__(self, fanova):
        self._fanova = fanova

    def create_all_plots(self, directory, **kwargs):
        """
            Create plots for all main effects.
        """
        assert os.path.exists(directory), "directory %s doesn't exist" % directory

        # Categorical parameters
        for param_name in self._fanova.get_config_space().get_categorical_parameters():
            plt.clf()
            outfile_name = os.path.join(directory, param_name.replace(os.sep, "_") + ".png")
            print("creating %s" % outfile_name)
            self.plot_categorical_marginal(param_name)
            plt.savefig(outfile_name)

        # Continuous and integer parameters
        params_to_plot = []
        params_to_plot.extend(self._fanova.get_config_space().get_continuous_parameters())
        params_to_plot.extend(self._fanova.get_config_space().get_integer_parameters())
        for param_name in params_to_plot:
            plt.clf()
            outfile_name = os.path.join(directory, param_name.replace(os.sep, "_") + ".png")
            print("creating %s" % outfile_name)
            self.plot_marginal(param_name, **kwargs)
            plt.savefig(outfile_name)

    def create_most_important_pairwise_marginal_plots(self, directory, n=20):
        categorical_parameters = self._fanova.get_config_space().get_categorical_parameters()

        most_important_pairwise_marginals = self._fanova.get_most_important_pairwise_marginals(n)
        for param1, param2 in most_important_pairwise_marginals:
            if param1 in categorical_parameters or param2 in categorical_parameters:
                print("skipping pairwise marginal plot %s x %s, because one of them is categorical" % (param1, param2))
                continue
            outfile_name = os.path.join(directory, param1.replace(os.sep, "_") + "x" + param2.replace(os.sep, "_") + ".png")
            plt.clf()
            print("creating %s" % outfile_name)
            self.plot_pairwise_marginal(param1, param2)
            plt.savefig(outfile_name)

    # TODO: Add kwargs to control plot presentatio
    def plot_categorical_marginal(self, param, ax=None):
        """
        Plot a marginal from a categorical hyperparameter
        :param param: str. Name of the parameter to be plotted. Must be categorical
        :param ax: Optional. If provided plot on this axis
        :return ax: matplotlib Axes. Returns Axes object for further tweaking.
        """
        if isinstance(param, int):
            dim = param
            param_name = self._fanova.get_parameter_names()[dim]
        else:
            if param not in self._fanova.param_name2dmin:
                print("Parameter %s not known" % param)
                return
            dim = self._fanova.param_name2dmin[param]
            param_name = param

        if param_name not in self._fanova.get_config_space().get_categorical_parameters():
            raise ValueError("Parameter %s is not a categorical parameter!" % param_name)

        categorical_size = self._fanova.get_config_space().get_categorical_size(param_name)

        labels = self._fanova.get_config_space().get_categorical_values(param)

        if ax is None:
            ax = plt.gca()

        indices = np.array(range(categorical_size))+1
        marginals = [self._fanova.get_categorical_marginal_for_value(param_name, i) for i in range(categorical_size)]
        mean, std = list(zip(*marginals))  # Collect all means and std values
        ax.errorbar(indices, mean, yerr=std, marker='s',
                    elinewidth=1.75, capsize=7.0, color='b',
                    ms=10.0, mec='white', mew=2.25, ls='dotted', lw=2.25)
        ax.set_xlim(indices[0]-0.3, indices[-1]+0.3)
        ax.set_xticks(indices)
        ax.set_xticklabels(labels)

        ax.set_ylabel("Performance")
        ax.set_xlabel(param_name)

        return ax

    def _check_param(self, param):
        if isinstance(param, int):
            dim = param
            param_name = self._fanova.get_parameter_names()[dim]
        else:
            assert param in self._fanova.param_name2dmin, "param %s not known" % param
            dim = self._fanova.param_name2dmin[param]
            param_name = param

        return dim, param_name

    def plot_pairwise_marginal(self, param_1, param_2, lower_bound_1=0, upper_bound_1=1, lower_bound_2=0, upper_bound_2=1, resolution=20):

        dim1, param_name_1 = self._check_param(param_1)
        dim2, param_name_2 = self._check_param(param_2)

        grid_1 = np.linspace(lower_bound_1, upper_bound_1, resolution)
        grid_2 = np.linspace(lower_bound_2, upper_bound_2, resolution)

        zz = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_2):
            for j, x_value in enumerate(grid_1):
                zz[i * resolution + j] = self._fanova._get_marginal_for_value_pair(dim1, dim2, x_value, y_value)[0]

        zz = np.reshape(zz, [resolution, resolution])

        display_grid_1 = [self._fanova.unormalize_value(param_name_1, value) for value in grid_1]
        display_grid_2 = [self._fanova.unormalize_value(param_name_2, value) for value in grid_2]

        display_xx, display_yy = np.meshgrid(display_grid_1, display_grid_2)

        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = Axes3D(fig)

        surface = ax.plot_surface(display_xx, display_yy, zz, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel(param_name_1)
        ax.set_ylabel(param_name_2)
        ax.set_zlabel("Performance")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        return plt

    def plot_categorical_pairwise(self, param_categorical, param_continuous, bound=(0, 1),
                                  log_scale=False, resolution=100,
                                  ax=None):
        """
        Plot the interaction of one categorical and one continuous or integer marginals
        :param param_categorical: String. Name of the first marginal. Must be categorical type parameter.
        :param param_continuous: String. Name of the second marginal. Must be integer or continuous type parameter.
        :param bound: Tuple. Limits or bounds of plotting of the continuous parameter.
        :param log_scale: Boolean. Wheter the axis of the continuous parameter is in log scale
        :param resolution: Scalar. Number of steps to predict the value of the continuos marginal
        :param ax: Optional. If provided plot on this axis
        :return ax: matplotlib Axes. Returns Axes object for further tweaking.
        """
        from itertools import chain, cycle
        import matplotlib.font_manager as fontmanager

        dim_inx1, param_name_1 = self._check_param(param_categorical)
        dim_inx2, param_name_2 = self._check_param(param_continuous)

        list_integers = self._fanova.get_config_space().get_integer_parameters()
        list_continuous = self._fanova.get_config_space().get_continuous_parameters()
        list_categorical = self._fanova.get_config_space().get_categorical_parameters()

        if param_name_1 in chain(list_integers, list_continuous):
            raise ValueError("Parameter %s cannot be continuous or integer parameter!", param_categorical)
        if param_name_2 in list_categorical:
            raise ValueError("Parameter %s cannot be categorical parameter!", param_continuous)

        # Prepare the canvas to plot
        if ax is None:
            ax = plt.gca()

        # Start with the preparations for continuous plotting
        grid = np.linspace(bound[0], bound[1], resolution)
        display_grid = [self._fanova.unormalize_value(param_name_2, value) for value in grid]

        # Get the different values for categorical parameters
        categorical_size = self._fanova.get_config_space().get_categorical_size(param_name_1)
        labels = self._fanova.get_config_space().get_categorical_values(param_name_1)

        # TODO: Add a seborn palette or better yet import seaborn
        codes = 'bgrmyck'
        color_codes = cycle(codes)

        if log_scale or (np.diff(display_grid).std() > 0.000001):
            canva = ax.semilogx
        else:
            canva = ax.plot

        for l in range(categorical_size):
            mean = np.zeros(resolution)
            std = np.zeros(resolution)
            for i in range(0, resolution):
                (m, s) = self._fanova._get_marginal_for_value_pair(param1=param_name_1, param2=param_name_2,
                                                                   value1=l, value2=grid[i])
                mean[i] = m
                std[i] = s
            mean = np.asarray(mean)
            std = np.asarray(std)
            colr = next(color_codes)
            canva(display_grid, mean, color=colr, label=labels[l])
            ax.fill_between(display_grid, mean+std, mean-std, facecolor=colr, alpha=0.3)
            lgs = ax.legend(loc='best', ncol=2, title=param_name_1, fontsize=10)
            lgs.get_title().set_size(10)
            ax.set_xlabel(param_name_2)
            ax.set_ylabel("Performance")
        return ax

    # TODO: Add kwargs to control plot presentation
    def plot_contour_pairwise(self, param_1, param_2,
                              bounds_1=(0, 1),
                              bounds_2=(0, 1),
                              log_scale_1=False, log_scale_2=False,
                              resolution=20, ax=None):
        """
        Plot the contour of interaction of two continuous or integers marginals
        Due to inability to detect log conditioning, must be passed explicitly
        :param param_1: String. Name of the first marginal. Must be integer or continuous type parameter.
        :param param_2: String. Name of the second marginal. Must be integer or continuous type parameter.
        :param bounds_1:
        :param bounds_2:
        :param log_scale_1:
        :param log_scale_2:
        :param resolution:
        :param ax: Optional. If provided plot on this axis
        :return ax: matplotlib Axes. Returns Axes object for further tweaking.
        """
        dim1, param_name_1 = self._check_param(param_1)
        dim2, param_name_2 = self._check_param(param_2)

        grid_1 = np.linspace(bounds_1[0], bounds_1[1], resolution)
        grid_2 = np.linspace(bounds_2[0], bounds_1[1], resolution)

        Z = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_2):
            for j, x_value in enumerate(grid_1):
                Z[i * resolution + j] = self._fanova._get_marginal_for_value_pair(dim1, dim2, x_value, y_value)[0]

        Z = np.reshape(Z, [resolution, resolution])

        display_grid_1 = [self._fanova.unormalize_value(param_name_1, value) for value in grid_1]
        display_grid_2 = [self._fanova.unormalize_value(param_name_2, value) for value in grid_2]

        X, Y = np.meshgrid(display_grid_1, display_grid_2)

        if ax is None:
            ax = plt.gca()

        contour_surface = ax.contourf(X, Y, Z, cmap='jet_r')
        ax.contour(contour_surface, color='k')
        if log_scale_1:
            ax.set_xscale('log')
        if log_scale_2:
            ax.set_yscale('log')
        ax.set_xlabel(param_name_1)
        ax.set_ylabel(param_name_2)
        cbar = ax.figure.colorbar(contour_surface, ax=ax)
        cbar.ax.set_ylabel('Performance')
        return ax

    def plot_3dcontour(self, param_1, param_2,
                       bounds_1=(0, 1), bounds_2=(0, 1),
                       log_scale_1=False, log_scale_2=False,
                       resolution=20):
        """
        Plot the contour in 3d projection of interaction of two continuous or integers marginals
        Due to inability to detect log conditioning, must be passed explicitly
        :param param_1: String. Name of the first marginal. Must be integer or continuous type parameter.
        :param param_2: String. Name of the second marginal. Must be integer or continuous type parameter.
        :param bounds_1:
        :param bounds_2:
        :param log_scale_1:
        :param log_scale_2:
        :param resolution:
        :return ax: matplotlib Axes. Returns Axes object for further tweaking.
        """
        dim1, param_name_1 = self._check_param(param_1)
        dim2, param_name_2 = self._check_param(param_2)

        grid_1 = np.linspace(bounds_1[0], bounds_1[1], resolution)
        grid_2 = np.linspace(bounds_2[0], bounds_1[1], resolution)

        Z = np.zeros([resolution * resolution])
        for i, y_value in enumerate(grid_2):
            for j, x_value in enumerate(grid_1):
                Z[i * resolution + j] = self._fanova._get_marginal_for_value_pair(dim1, dim2, x_value, y_value)[0]

        Z = np.reshape(Z, [resolution, resolution])

        display_grid_1 = [self._fanova.unormalize_value(param_name_1, value) for value in grid_1]
        display_grid_2 = [self._fanova.unormalize_value(param_name_2, value) for value in grid_2]

        X, Y = np.meshgrid(display_grid_1, display_grid_2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.6, cmap='jet_r')
        contour_surface = ax.contourf(X, Y, Z, cmap='jet_r')
        #contour_surface_1 = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='jet_r')
        #contour_surface_2 = ax.contourf(X, Y, Z, zdir='x', offset=0.0, cmap='Blues')
        #contour_surface_3 = ax.contourf(X, Y, Z, zdir='y', offset=0.01, cmap='Blues')

        if log_scale_1:
            ax.set_xscale('log')
        if log_scale_2:
            ax.set_yscale('log')
        ax.set_xlabel(param_name_1)
        ax.set_ylabel(param_name_2)
        return ax

    # TODO: Add kwargs to control plot presentation
    def plot_marginal(self, param, lower_bound=0, upper_bound=1,
                      is_int=False, resolution=100, log_scale=False,
                      ax=None):
        if isinstance(param, int):
            dim = param
            param_name = self._fanova.get_parameter_names()[dim]
        else:
            if param not in self._fanova.param_name2dmin:
                print("Parameter %s not known" % param)
                return
            dim = self._fanova.param_name2dmin[param]
            param_name = param

        # TODO: Validate that if not categorical then plot the categorical parameter!!!
        if param_name not in self._fanova.get_config_space().get_integer_parameters() and \
           param_name not in self._fanova.get_config_space().get_continuous_parameters():
            print("Parameter %s is not a continuous or integer parameter!" % param_name)
            return 
        grid = np.linspace(lower_bound, upper_bound, resolution)
        display_grid = [self._fanova.unormalize_value(param_name, value) for value in grid]

        mean = np.zeros(resolution)
        std = np.zeros(resolution)
        for i in range(0, resolution):
            (m, s) = self._fanova.get_marginal_for_value(dim, grid[i])
            mean[i] = m
            std[i] = s
        mean = np.asarray(mean)
        std = np.asarray(std)

        lower_curve = mean - std
        upper_curve = mean + std

        if ax is None:
            ax = plt.gca()

        if log_scale or (np.diff(display_grid).std() > 0.000001 and param_name in self._fanova.get_config_space().get_continuous_parameters()):
            #HACK for detecting whether it's a log parameter, because the config space doesn't expose this information
            ax.semilogx(display_grid, mean, 'b')
            #print "printing %s semilogx" % param_name
        else:
            ax.plot(display_grid, mean, 'b')
        ax.fill_between(display_grid, upper_curve, lower_curve, facecolor='b', alpha=0.3)
        ax.set_xlabel(param_name)

        ax.set_ylabel("Performance")
        return ax
    
#     def create_pdf_file(self):
#         latex_doc = self._latex_template
#         with open("fanova_output.tex", "w") as fh:
#             fh.write(latex_doc)        
#             subprocess.call('pdflatex fanova_output.tex')
