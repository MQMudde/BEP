import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import csv


class ParametersFromSampleFile:

    def __init__(self, file):
        # drop the extension
        filepath, _ = os.path.splitext(file)
        # drop the folder(s) name
        filename = os.path.basename(filepath)
        split_string = filename.split()
        self.sampletype = split_string[0]
        self.xi = float(split_string[1])
        self.samplesize = int(split_string[3])
        self.nrsamples = int(split_string[5])
        self.noisetype = split_string[7]
        if self.noisetype != 'None':
            self.noiseparam = split_string[8]
            if len(split_string) == 10:
                self.result_type = split_string[9]
        else:
            if len(split_string) == 9:
                self.result_type = split_string[8]


def read_single_line_file(file):
    with open(file, 'r', newline='') as measurements:
        reader = csv.reader(measurements, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for line in reader:
            return line


def histogram_plotter(ax, data, param_dict):
    nr_bins = 50
    binwidth = (max(data) - min(data)) / nr_bins
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    out = ax.hist(data, bins=bins, density=True, **param_dict)
    return out


def line_plotter(ax, data, param_dict):
    out = sns.lineplot(x=range(1, len(data) + 1), y=data, ax=ax, **param_dict)
    return out


# takes data in dictionary form key: value - x: y
def loglog_plotter(ax, data_dict, param_dict):
    out = sns.lineplot(x=data_dict.keys(), y=data_dict.values(), ax=ax, **param_dict)
    ax.set_xscale('log')
    ax.set_yscale('log')
    return out


def convergence_plotter(ax, data_dict, name):
    ax.set_title(f'MSE of Hill estimator {name}')
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    return loglog_plotter(ax, data_dict, {})


def plot_convergence(data_dict):
    fig, ax = plt.subplots(figsize=(10, 10))
    for result_type in data_dict:
        sns.lineplot(x=data_dict[result_type].keys(), y=data_dict[result_type].values(), ax=ax, label=result_type)
    ax.set_title('MSE of Hill estimator')
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig


def plot_from_file(ax, file):
    param_container = ParametersFromSampleFile(file)
    sample_size = param_container.samplesize
    file_type = param_container.result_type
    data = read_single_line_file(file)
    dict_types = \
        {'Variance': {'title': 'Distribution of the variance of the Hill estimator', 'xlabel': 'variance'},
         'Distance': {'title': 'Distribution of the distance between the Hill estimator of two samples',
                      'xlabel': 'distance'},
         'WaldIntervals': {'title': 'Average Wald confidence interval size', 'xlabel': 'k', 'ylabel': 'size',
                           'ylim': (0, 1), 'xlim': (0, sample_size)},
         'ScoreIntervals': {'title': 'Average Score confidence interval size', 'xlabel': 'k', 'ylabel': 'size',
                            'ylim': (0, 1), 'xlim': (0, sample_size)},
         'LRIntervals': {'title': 'Average LR confidence interval size', 'xlabel': 'k', 'ylabel': 'size',
                         'ylim': (0, 1), 'xlim': (0, sample_size)},
         'BCLRIntervals': {'title': 'Average BCLR confidence interval size', 'xlabel': 'k', 'ylabel': 'size',
                           'ylim': (0, 1), 'xlim': (0, sample_size)},
         'WaldRejection': {'title': 'Wald confidence interval rejection probability', 'xlabel': 'k',
                           'ylabel': 'rejection probability', 'ylim': (0, 1), 'xlim': (0, sample_size)},
         'ScoreRejection': {'title': 'Score confidence interval rejection probability', 'xlabel': 'k',
                            'ylabel': 'rejection probability', 'ylim': (0, 1), 'xlim': (0, sample_size)},
         'LRRejection': {'title': 'LR confidence interval rejection probability', 'xlabel': 'k',
                         'ylabel': 'rejection probability', 'ylim': (0, 1), 'xlim': (0, sample_size)},
         'BCLRRejection': {'title': 'BCLR confidence interval rejection probability', 'xlabel': 'k',
                           'ylabel': 'rejection probability', 'ylim': (0, 1), 'xlim': (0, sample_size)},
         'MSE': {'title': 'MSE of the Hill estimator', 'xlabel': 'k', 'ylabel': 'MSE', 'xlim': (0, sample_size)}
         }
    try:
        ax.set_title(dict_types[file_type]['title'])
        ax.set_xlabel(dict_types[file_type]['xlabel'], fontsize=12)
        ax.set_ylabel(dict_types[file_type]['ylabel'], fontsize=12)
    except:
        pass
    try:
        ax.set_ylim(dict_types[file_type]['ylim'])
    except:
        pass
    try:
        ax.set_xlim(dict_types[file_type]['xlim'])
    except:
        pass
    if file_type in {'WaldRejection', 'ScoreRejection', 'LRRejection', 'BCLRRejection'}:
        ax.axhline(0.1, color='Black', linewidth=0.4)
    elif file_type in {'Distance', 'Variance'}:
        return histogram_plotter(ax, data, {})
    return line_plotter(ax, data, {})


# TODO make this function working with 1D axs and make it remove labels from axes as well
def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
            ax.set_xlabel('')
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
    if sharex and axs.ndim == 1:
        for ax in axs[:-1].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
            ax.set_xlabel('')


def plot_results_file_list(nr_rows, nr_columns, measurement_files_list, fig_size, dpi, title=None):
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(nr_rows, nr_columns, figsize=fig_size, dpi=dpi)
    for index, measurement_file in enumerate(measurement_files_list):
        ax = fig.axes[index]
        plot_from_file(ax, measurement_file)
    if title is not None:
        fig.suptitle(title)
    return fig


def normed_plot(ax, data, param_dict):
    sample_size = len(data) + 1
    x_points = np.linspace(0, 1, num=sample_size)[1:]
    return sns.lineplot(x=x_points, y=data, ax=ax, **param_dict)


def add_plot_style(func):
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        ax = kwargs['ax']
        res_type = kwargs['res_type']
        set_log_axes = kwargs['set_log_axes']
        ax.legend()
        if res_type in {'WaldRejection', 'ScoreRejection', 'LRRejection', 'BCLRRejection'}:
            ax.axhline(0.1, color='Black', linewidth=0.4)
            ax.set_ylim((0, 1))
        # x-label and x-limits
        if res_type not in {'Variance', 'Distance'}:
            ax.set_xlabel('$k/n$', fontsize=12, labelpad=0)
            if set_log_axes[0]:
                ax.set_xlim(1 / sample_size, 1)
            else:
                ax.set_xlim(0, 1)
        else:
            ax.set_xlabel(res_type, fontsize=12, labelpad=0)
        # logarithmic axes
        if set_log_axes[0]:
            ax.set_xscale('log')
        if set_log_axes[1]:
            ax.set_yscale('log')
        # y-label
        y_label_dict = {
            'Variance': 'density',
            'Distance': 'density',
            'MSE': 'MSE',
            'WaldIntervals': 'size',
            'ScoreIntervals': 'size',
            'LRIntervals': 'size',
            'BCLRIntervals': 'size',
            'WaldRejection': 'rej. prob.',
            'ScoreRejection': 'rej. prob.',
            'LRRejection': 'rej. prob.',
            'BCLRRejection': 'rej. prob.',
        }
        ax.set_ylabel(y_label_dict[res_type], fontsize=12)
    return inner


@add_plot_style
def plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='None', show_xi_type_dict={}, ax=None, set_log_axes=(False, False)):
    for xi, bool in show_xi_type_dict.items():
        if bool:
            if noise_type == 'None':
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {res_type}.csv'
            else:
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {noise_param} {res_type}.csv'
            measurement = read_single_line_file(file)
            if res_type in {'Variance', 'Distance'}:
                histogram_plotter(ax, measurement, {'label': f'$\\xi = {xi}$', 'alpha': 0.7})
            else:
                normed_plot(ax, measurement, {'label': f'$\\xi = {xi}$'})


@add_plot_style
def plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='None', show_sample_size_dict={}, ax=None, set_log_axes=(False, False)):
    for sample_size, bool in show_sample_size_dict.items():
        if bool:
            if noise_type == 'None':
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {res_type}.csv'
            else:
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {noise_param} {res_type}.csv'
            measurement = read_single_line_file(file)
            if res_type in {'Variance', 'Distance'}:
                avg = np.mean(measurement)
                print(f'avg {res_type} for n {sample_size}: {avg}')
                histogram_plotter(ax, measurement, {'label': f'$n = {len(measurement) + 1}$', 'alpha': 0.7})
            else:
                normed_plot(ax, measurement, {'label': f'$n = {len(measurement) + 1}$'})


@add_plot_style
def plot_results_all_noise_types(sample_type, xi, sample_size, res_type='None', show_noise_type_dict={}, ax=None, set_log_axes=(False, False)):
    for noise_type, bool in show_noise_type_dict.items():
        if bool:
            file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {res_type}.csv'
            measurement = read_single_line_file(file)
            label_dict = {
                'None': 'No noise',
                'Gaussian lower0upper1': '$\\mathcal{N}[0,1]$',
                'Gaussian lower-0.5upper0.5': '$\\mathcal{N}[-0.5,0.5]$',
                'Uniform loc-1scale1': '$\\mathrm{U}[-1,0]$',
                'Uniform loc-0.5scale1': '$\\mathrm{U}[-0.5,0.5]$',
                'Uniform loc0scale1': '$\\mathrm{U}[0,1]$',
                'Pareto xi0.5': 'Par. $\\xi=0.5$',
                'Pareto xi0.6': 'Par. $\\xi=0.6$',
                'Pareto xi0.7': 'Par. $\\xi=0.7$',
                'Pareto xi0.8': 'Par. $\\xi=0.8$',
                'Gaussian lower-1upper0': '$\\mathcal{N}[-1,0]$',
                'Uniform loc-0.5scale0.5': '$\\mathrm{U}[-0.5,0]$',
                'Uniform loc-0.25scale0.5': '$\\mathrm{U}[-0.25,0.25]$',
                'Uniform loc0scale0.5': '$\\mathrm{U}[0,0.5]$'
            }
            if res_type in {'Variance', 'Distance'}:
                histogram_plotter(ax, measurement, {'label': label_dict[noise_type], 'alpha': 0.7})
            else:
                normed_plot(ax, measurement, {'label': label_dict[noise_type]})


@add_plot_style
def plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='None', show_sample_type_dict={}, ax=None, set_log_axes=(False, False)):
    for sample_type, bool in show_sample_type_dict.items():
        if bool:
            if noise_type == 'None':
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {res_type}.csv'
            else:
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {noise_param} {res_type}.csv'
            measurement = read_single_line_file(file)
            label_dict = {
                'RoundedPareto': 'Floored Pareto',
                'MixedPoisson': 'Poisson mixture',
                'Zipf': 'Zipf',
                'PreferentialAttachment': 'Preferential Attachment'
            }
            if res_type in {'Variance', 'Distance'}:
                histogram_plotter(ax, measurement, {'label': label_dict[sample_type], 'alpha': 0.7})
            else:
                normed_plot(ax, measurement, {'label': label_dict[sample_type]})


# The following function adds various result types to the same ax object.
# Therefore using this only makes sense when comparing results of the same type. E.g. comparing BCLRIntervals and LRIntervals.
@add_plot_style
def plot_results_all_result_types(sample_type, xi, sample_size, noise_type, noise_param, res_type='None', show_res_type_dict={}, ax=None, set_log_axes=(False, False)):
    for res_type, bool in show_res_type_dict.items():
        if bool:
            if noise_type == 'None':
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {res_type}.csv'
            else:
                file = f'../Data/{sample_type} {xi} xi {sample_size} n 10000 samples/{sample_type} {xi} xi {sample_size} n 10000 samples {noise_type} {noise_param} {res_type}.csv'
            measurement = read_single_line_file(file)
            label_dict = {
                'WaldIntervals': 'Wald',
                'ScoreIntervals': 'Score',
                'LRIntervals': 'LR',
                'BCLRIntervals': 'BCLR',
                'WaldRejection': 'Wald',
                'ScoreRejection': 'Score',
                'LRRejection': 'LR',
                'BCLRRejection': 'BCLR'
            }
            normed_plot(ax, measurement, {'label': label_dict[res_type]})


if __name__ == "__main__":
    sample_type = 'PreferentialAttachment'
    xi = 0.5
    noise_type = 'Uniform'
    noise_param = 'loc-0.5scale1'
    sample_size = 10000

    sns.set_style('whitegrid')

    # xi variation results plot
    show_xi_type_dict = {
        0.5: True,
        0.6: True,
        0.7: True,
        0.8: True
    }
    # fig, axs = plt.subplots(5, 1, figsize=(8,4), dpi=96)
    # plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='MSE',
    #                         show_xi_type_dict=show_xi_type_dict, ax=axs[0], set_log_axes=(False, True))
    # plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='BCLRIntervals',
    #                         show_xi_type_dict=show_xi_type_dict, ax=axs[1], set_log_axes=(False, True))
    # plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='BCLRRejection',
    #                         show_xi_type_dict=show_xi_type_dict, ax=axs[2], set_log_axes=(False, False))
    # plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='Variance',
    #                         show_xi_type_dict=show_xi_type_dict, ax=axs[3], set_log_axes=(False, False))
    # plot_results_all_xi_types(sample_type, sample_size, noise_type, noise_param, res_type='Distance',
    #                         show_xi_type_dict=show_xi_type_dict, ax=axs[4], set_log_axes=(False, False))
    # set_share_axes(axs[:3], sharex=True, sharey=False)
    # axs[1].get_legend().remove()
    # axs[2].get_legend().remove()
    # axs[3].get_legend().remove()
    # axs[4].get_legend().remove()
    # plt.subplots_adjust(left=0.07, bottom=0.04, right=0.988, top=0.99, hspace=0.27)

    # Sample size variation results plot
    show_size_dict = {100: True,
                      200: True,
                      500: True,
                      1000: True,
                      2000: True,
                      5000: True,
                      10000: True
                      }
    # fig, axs = plt.subplots(5, 1, figsize=(8, 4), dpi=96)
    # plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='MSE',
    #                         show_sample_size_dict=show_size_dict, ax=axs[0], set_log_axes=(True, True))
    # plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='BCLRIntervals',
    #                         show_sample_size_dict=show_size_dict, ax=axs[1], set_log_axes=(True, True))
    # plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='BCLRRejection',
    #                         show_sample_size_dict=show_size_dict, ax=axs[2], set_log_axes=(True, False))
    # plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='Variance',
    #                         show_sample_size_dict=show_size_dict, ax=axs[3], set_log_axes=(False, False))
    # plot_results_all_sample_sizes(sample_type, xi, noise_type, noise_param, res_type='Distance',
    #                         show_sample_size_dict=show_size_dict, ax=axs[4], set_log_axes=(True, False))
    # set_share_axes(axs[:3], sharex=True, sharey=False)
    # axs[1].get_legend().remove()
    # axs[2].get_legend().remove()
    # axs[3].get_legend().remove()
    # axs[4].get_legend().remove()
    # plt.subplots_adjust(left=0.07, bottom=0.04, right=0.988, top=0.99, hspace=0.27)

    # Noise type variation results plot
    show_noise_type_dict = {
        'None': False,
        'Gaussian lower0upper1': False, #
        'Gaussian lower-0.5upper0.5': False,
        'Uniform loc-1scale1': False,
        'Uniform loc-0.5scale1': False,
        'Uniform loc0scale1': False,
        'Pareto xi0.5': False, #
        'Pareto xi0.6': False,
        'Pareto xi0.7': False,
        'Pareto xi0.8': False,
        'Gaussian lower-1upper0': False,
        'Uniform loc-0.5scale0.5': False,
        'Uniform loc-0.25scale0.5': False,
        'Uniform loc0scale0.5': False
    }
    # fig, axs = plt.subplots(4, 1, figsize=(8, 4), dpi=96)
    # plot_results_all_noise_types(sample_type, xi, sample_size, res_type='MSE',
    #                              show_noise_type_dict=show_noise_type_dict, ax=axs[0], set_log_axes=(False, True))
    # plot_results_all_noise_types(sample_type, xi, sample_size, res_type='BCLRIntervals',
    #                              show_noise_type_dict=show_noise_type_dict, ax=axs[1], set_log_axes=(False, True))
    # plot_results_all_noise_types(sample_type, xi, sample_size, res_type='BCLRRejection',
    #                              show_noise_type_dict=show_noise_type_dict, ax=axs[2], set_log_axes=(False, False))
    # plot_results_all_noise_types(sample_type, xi, sample_size, res_type='Variance',
    #                              show_noise_type_dict=show_noise_type_dict, ax=axs[3], set_log_axes=(False, False))
    # # plot_results_all_noise_types(sample_type, xi, sample_size, res_type='Distance',
    # #                              show_noise_type_dict=show_noise_type_dict, ax=axs[4], set_log_axes=(False, False))
    # set_share_axes(axs[:3], sharex=True, sharey=False)
    # axs[1].get_legend().remove()
    # axs[2].get_legend().remove()
    # axs[3].get_legend().remove()
    # #axs[4].get_legend().remove()
    # plt.subplots_adjust(left=0.07, bottom=0.04, right=0.988, top=0.99, hspace=0.27)

    # Sample type variation results plot
    show_sample_type_dict = {
        'RoundedPareto': True,
        'MixedPoisson': True,
        'Zipf': True,
        'PreferentialAttachment': True
    }
    # fig, axs = plt.subplots(5, 1, figsize=(8, 4), dpi=96)
    # plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='MSE',
    #                               show_sample_type_dict=show_sample_type_dict, ax=axs[0], set_log_axes=(False, True))
    # plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='BCLRIntervals',
    #                               show_sample_type_dict=show_sample_type_dict, ax=axs[1], set_log_axes=(False, True))
    # plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='BCLRRejection',
    #                               show_sample_type_dict=show_sample_type_dict, ax=axs[2], set_log_axes=(False, False))
    # plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='Variance',
    #                               show_sample_type_dict=show_sample_type_dict, ax=axs[3], set_log_axes=(False, False))
    # plot_results_all_sample_types(xi, sample_size, noise_type, noise_param, res_type='Distance',
    #                               show_sample_type_dict=show_sample_type_dict, ax=axs[4], set_log_axes=(False, False))
    # set_share_axes(axs[:3], sharex=True, sharey=False)
    # axs[0].get_legend().remove()
    # axs[1].get_legend().remove()
    # axs[2].get_legend().remove()
    # axs[3].get_legend().remove()
    # axs[4].get_legend().remove()
    # plt.subplots_adjust(left=0.07, bottom=0.04, right=0.988, top=0.99, hspace=0.27)

    # Result type variation results plot. Useful for comparing confidence intervals.
    show_res_type_dict_intervals = {
        'WaldIntervals': True,
        'ScoreIntervals': True,
        'LRIntervals': True,
        'BCLRIntervals': True
    }
    show_res_type_dict_rejection = {
        'WaldRejection': True,
        'ScoreRejection': True,
        'LRRejection': True,
        'BCLRRejection': True
    }
    # fig, axs = plt.subplots(2, 1, figsize=(10, 4.7), dpi=96)
    # plot_results_all_result_types(sample_type, xi, sample_size, noise_type, noise_param, res_type='WaldIntervals',
    #                               show_res_type_dict=show_res_type_dict_intervals, ax=axs[0], set_log_axes=(False, True))
    # plot_results_all_result_types(sample_type, xi, sample_size, noise_type, noise_param, res_type='WaldRejection',
    #                               show_res_type_dict=show_res_type_dict_rejection, ax=axs[1],
    #                               set_log_axes=(False, False))
    # set_share_axes(axs, sharex=True, sharey=False)
    # axs[1].get_legend().remove()
    # plt.subplots_adjust(left=0.07, bottom=0.11, right=0.988, top=0.99, hspace=0.27)


    plt.show()