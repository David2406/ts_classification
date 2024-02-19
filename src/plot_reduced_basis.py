import matplotlib.pyplot as plt
import numpy as np


def plot_cumulated_variance(base,
                            number_of_modes = None,
                            x_label = 'Number of modes',
                            y_label = 'Cumulated variance',
                            x_ticks = None,
                            y_ticks = None,
                            legend_loc = 'lower right',
                            title = 'Time series of variable {}',
                            linestyle = '-',
                            plot_style = 'o',
                            grid = True,
                            fig_size = (10, 8),
                            save_filename = "cumulated_variance_{}.pdf"):
    """
    Parameters:
    -----------
    Returns:
    --------
    """
    cumulated_variance = base.singular.cumsum() / base.singular.sum()
    plt.figure(figsize = fig_size)
    if number_of_modes is not None:
        plt.plot(range(1, number_of_modes + 1),
                 cumulated_variance[:number_of_modes],
                 linestyle = linestyle,
                 marker = plot_style)
    else:
        plt.plot(cumulated_variance,
                 linestyle = linestyle,
                 marker = plot_style,
                 )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(grid)
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename.format(number_of_modes))
    else:
        plt.show()
    plt.close()


def plot_reduced_mode(base,
                      mode = 1,
                      x_vals = np.arange(0, 10, 1),
                      x_label = 'Time (s)',
                      y_label = 'Mode value',
                      x_ticks = None,
                      y_ticks = None,
                      legend_loc = 'lower right',
                      title = 'Reduced basis mode {}',
                      linestyle = '-',
                      plot_style = 'o',
                      markersize = 0.5,
                      grid = True,
                      fig_size = (10, 8),
                      save_filename = "reduced_mode_{}.pdf"):
    """
    Plots a mode from a reduced basis.

    Parameters:
    -----------
    base: ReducedBasis
        The reduced basis.
    mode: int
        The mode we want to plot. ATTENTION: the user indicates the mode in a range from 1 (the first mode) to the last.
        In the reduced basis they are numbered starting from 0.

    Returns:
    --------
    """
    plt.figure(figsize = fig_size)
    plt.plot(x_vals,
             base.base[:, mode - 1],
             linestyle = linestyle,
             marker = plot_style,
             markersize = markersize,
             )
    plt.title(title.format(mode))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(grid)
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename.format(mode))
    else:
        plt.show()
    plt.close()


def plot_reduced_modes_together(base,
                                x_vals,
                                modes = [0, 1, 2, 3],
                                x_label = 'Time (s)',
                                y_label = 'Mode value',
                                x_ticks = None,
                                y_ticks = None,
                                legend_loc = 'lower right',
                                title = 'Reduced basis modes {}, {}, {}, {}',
                                linestyle = '-',
                                plot_style = 'o',
                                markersize = 0.5,
                                grid = True,
                                fig_size = (10, 8),
                                save_filename = "reduced_modes_{}_{}_{}_{}.pdf"):
    """
    Plots a group of reduced modes from a reduced basis in a (2, 2) frame.

    Parameters:
    -----------
    base: ReducedBasis
        The reduced basis.
    mode: list
        The modes we want to plot. ATTENTION: the user indicates the mode in a range starting from 1 (the first mode).
        In the reduced basis they are numbered starting from 0.

    Returns:
    --------
    """
    fig, axs = plt.subplots(2, 2, figsize = fig_size)
    fig.suptitle(title.format(*modes))
    axs[0, 0].plot(x_vals,
                   base.base[:, modes[0] - 1],
                   linestyle = linestyle,
                   marker = plot_style,
                   markersize = markersize,
                   )
    axs[0, 0].set_title('Mode {}'.format(modes[0]))
    axs[0, 1].plot(x_vals,
                   base.base[:, modes[1] - 1],
                   linestyle = linestyle,
                   marker = plot_style,
                   markersize = markersize, )
    axs[0, 1].set_title('Mode {}'.format(modes[1]))
    axs[1, 0].plot(x_vals,
                   base.base[:, modes[2] - 1],
                   linestyle = linestyle,
                   marker = plot_style,
                   markersize = markersize, )
    axs[1, 0].set_title('Mode {}'.format(modes[2]))
    axs[1, 1].plot(x_vals,
                   base.base[:, modes[3] - 1],
                   linestyle = linestyle,
                   marker = plot_style,
                   markersize = markersize, )
    axs[1, 1].set_title('Mode {}'.format(modes[3]))
    for ax in axs.flat:
        ax.set(xlabel = x_label)
        ax.set(ylabel = y_label)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(grid)
    fig.tight_layout()

    plt.savefig(save_filename.format(*modes))
    plt.close()


def plot_reconstruction(ref_result,
                        est_result,
                        x_values = None,
                        x_label = 'Time (s)',
                        y_label = 'Temperature (C)',
                        x_ticks = None,
                        y_ticks = None,
                        ref_legend = "Reference",
                        est_legend = "Reconstruction",
                        legend_loc = 'lower right',
                        title = 'Reference and reconstructed trajectories',
                        linestyle = '-',
                        plot_style = 'o',
                        markersize = 3,
                        grid = True,
                        fig_size = (10, 8),
                        savefile_name = "comparison_reconstruction_results.pdf"):

    if x_values is None:
        x_values = np.arange(ref_result.shape[0])

    plt.figure(figsize = fig_size)

    plt.plot(x_values,
             ref_result,
             linestyle = linestyle,
             marker = plot_style,
             markersize = markersize)
    plt.plot(x_values,
             est_result,
             linestyle = linestyle,
             marker = plot_style,
             markersize = markersize)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend([ref_legend, est_legend], loc = legend_loc)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig(savefile_name)
    plt.close()


def plot_rb_leads(mode_id,
                  Phi,
                  time_ts,
                  nb_pts = 5000,
                  nb_leads = 12,
                  lead_names = ['avf',
                                'avl',
                                'avr',
                                'i',
                                'ii',
                                'iii',
                                'v1',
                                'v2',
                                'v3',
                                'v4',
                                'v5',
                                'v6'],
                  nb_cols = 3):

    lead_modes = Phi[:, mode_id].reshape(nb_pts, nb_leads)
    nb_rows = 1 + int((nb_leads - 1) / nb_cols)
    title = 'Observation of mode {} time series'.format(mode_id + 1)

    fig, ax = plt.subplots(nb_rows, nb_cols)
    fig.suptitle(title)

    for lead_id, lead_name in enumerate(lead_names):
        i = lead_id % nb_cols
        j = int((lead_id - i) / nb_cols)
        lead_mode_ts = lead_modes[:, lead_id]

        ax[j, i].plot(time_ts,
                      lead_mode_ts)
        ax[j, i].title.set_text('Lead: {}'.format(lead_name))
        ax[j, i].xaxis.grid(True, which = 'major')
        ax[j, i].yaxis.grid(True, which = 'major')

    plt.show()
    plt.close()
