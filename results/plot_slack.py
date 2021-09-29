import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from utility_func import plot_poi, plot_ctrl_points, calc_ctrl_deriv, interpolate_b_spline

import pickle as pkl

def plot_c_converge(list_c):
    """
    Plot a time convergence of the control points as time increases
    """

    return

if __name__ == '__main__':
    filename_data = 'results/20200310_224433.pkl'
    dir_data = os.path.join( os.path.dirname(__file__), '..' )
    with open( os.path.join(dir_data, filename_data), 'rb') as fid:
        data = pkl.load(fid)

    # Restore data
    #   pickled data is of the form:
    #   {'vmax': vmax, 'amax': amax, 'jmax': jmax, 'B': B,
    #    'p_in': p_in, 'p_out': p_out, 'v_in': v_in, 'v_out': v_out,
    #    'slack': list_slack, 'c': list_c}

    colormap = plt.get_cmap('tab10')

    list_slack = data['slack']
    list_c = data['c']

    plt.ion()

    plt.figure(1, figsize=(6,6))
    ax_gamma = plt.subplot(111)

    list_legend = []
    ind_color = 0

    for ind_m, slack in enumerate(list_slack):
        if ind_m in [0, 2, 5, 7, 9]:
            color = colormap(ind_color)
            ax_gamma.plot(slack[:, 1], slack[:, 3], color=color)
            #ax_vaj.plot(slack[:, 1], slack[:, 3], color=color)
            # ax_vaj.plot(slack[:, 1], slack[:, 4], color=color, dashes=(2, 2, 1, 2, 1, 2))
            # ax_vaj.plot(slack[:, 1], slack[:, 6], color=color, dashes=(1, 1))
            # ax_vaj.plot(slack[:, 1], slack[:, 7], color=color)
            list_legend.append('M = {0:2d}'.format(int(slack[0,0])))
            ind_color += 1
    ind_color = 0
    for ind_m, slack in enumerate(list_slack):
        if ind_m in [0, 2, 5, 7, 9]:
            color = colormap(ind_color)
            ax_gamma.plot(slack[:, 1], slack[:, 5], color=color, dashes=(3, 2))
            ind_color += 1
    ax_gamma.legend(list_legend)
    ax_gamma.set_ylim([0.3, 5])
    ax_gamma.set_xlim([1.6, 4])
    ax_gamma.set_ylabel('Slack (%)', fontsize=14)
    ax_gamma.set_xlabel('Observation Time (s)', fontsize=14)
    ax_gamma.grid()
    plt.savefig('fig_slack_overview.png', transparent=True)
    plt.draw()
    plt.pause(0.01)
