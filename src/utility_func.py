import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

K_CONST = 3

"""
Helper functions
"""
def dist_l1(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sum(np.abs(x[0:ind_end] - y[0:ind_end]))


def dist_l2(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sqrt(np.sum(np.power(x[0:ind_end] - y[0:ind_end], 2)))

"""
Utility functions for calculating optimal trajectories using B-splines for order 3 functions

list_traj holds the data for the trajectories for each segment. Each element is a dict containing:
    'q': 
    't_valid': 
    't_res': 
    'tau': 
    'tau_p':  
    'c_p': 
    'tau_v': 
    'c_v': 
    'tau_a': 
    'c_a': 
    'tau_j': 
    'c_j': 
    't': 
    's_p': 
    's_v': 
    's_a': 
    's_j': 
"""

### Plotting functions
def plot_poi(list_traj):
    """
    Plot the points of interest and associated sensing regions
    """

    # Plot of 2D trajectory
    fig1 = plt.figure(1)
    ax_p = plt.subplot(111)
    x_prev = list_traj[-1]['q'].pos[0]
    y_prev = list_traj[-1]['q'].pos[1]
    for traj in list_traj:
        x, y = traj['q'].pos[0], traj['q'].pos[1]
        ax_p.plot([x_prev, x], [y_prev, y], 'k')
        ax_p.scatter(traj['q'].pos[0], traj['q'].pos[1])
        x_B, y_B = traj['q'].get_B_region()
        ax_p.plot(x_B, y_B, 'r:')
        x_prev = x
        y_prev = y

    for traj in list_traj:
        ax_p.plot(traj['s_p'][:, 0], traj['s_p'][:, 1])

    x_off, y_off = -20, -15
    ax_p.set_xlim([0 + x_off, 80 + x_off])
    ax_p.set_ylim([0 + y_off, 60 + y_off])
    plt.draw()
    plt.pause(0.001)

    # Time-series plot of velocity, acceleration, and jerk
    fig2 = plt.figure(2)
    ax_v_x = plt.subplot(231)
    ax_v_y = plt.subplot(234)
    ax_a_x = plt.subplot(232)
    ax_a_y = plt.subplot(235)
    ax_j_x = plt.subplot(233)
    ax_j_y = plt.subplot(236)

    t_off = 0
    for traj in list_traj:
        t_plt = traj['t'] + t_off

        ax_v_x.plot(t_plt, traj['s_v'][:, 0])
        ax_v_y.plot(t_plt, traj['s_v'][:, 1])

        ax_a_x.plot(t_plt, traj['s_a'][:, 0])
        ax_a_y.plot(t_plt, traj['s_a'][:, 1])

        ax_j_x.plot(t_plt, traj['s_j'][:, 0])
        ax_j_y.plot(t_plt, traj['s_j'][:, 1])

        t_off = t_plt[-1]

    ax_v_x.set_title('s_v(t)')
    ax_a_x.set_title('s_a(t)')
    ax_j_x.set_title('s_j(t)')
    ax_a_y.set_xlabel('Time (t)')
    plt.draw()
    plt.pause(0.001)

    return

def plot_ctrl_points(list_traj, vmax=10, amax=20, jmax=100):
    """
    Plots the control points for each trajectory segment
    """

    # Circle plot of control points
    v_max_test = vmax
    a_max_test = amax
    j_max_test = jmax

    fig3 = plt.figure(3, figsize=(12, 3.5))
    ax_crc_v = plt.subplot(131)
    ax_crc_a = plt.subplot(132)
    ax_crc_j = plt.subplot(133)

    theta = np.linspace(0, 2*np.pi, 100)
    x_scale = np.cos(theta)
    y_scale = np.sin(theta)
    ax_crc_v.plot(v_max_test * x_scale, v_max_test * y_scale, 'k')
    ax_crc_a.plot(a_max_test * x_scale, a_max_test * y_scale, 'k')
    ax_crc_j.plot(j_max_test * x_scale, j_max_test * y_scale, 'k')

    for traj in list_traj:
        ax_crc_v.scatter(traj['c_v'][:,0], traj['c_v'][:,1])
        ax_crc_a.scatter(traj['c_a'][:,0], traj['c_a'][:,1])
        ax_crc_j.scatter(traj['c_j'][:,0], traj['c_j'][:,1])

    ax_crc_v.set_title('c_v')
    ax_crc_a.set_title('c_a')
    ax_crc_j.set_title('c_j')
    plt.draw()
    plt.pause(0.001)

    return

### Calculating functions
def calc_ctrl_deriv(c_p, dt=1):
    """
    Calculate the control points c_d for d = 1 (v, velocity), 2 (a, acceleration), and 3 (j, jerk)

    NOTE: Due to convolution, points on the edge have invalid/irregular values. Use with caution.
    """
    A_c = [np.array([1]),
           np.array([1, -1]),
           np.array([1, -2, 1]),
           np.array([1, -3, 3, -1]),
           np.array([1, -4, 6, -4, 1])]

    c_v = np.empty(c_p.shape)
    c_a = np.empty(c_p.shape)
    c_j = np.empty(c_p.shape)

    # Calculate the control points c_d from c for the given derivative
    for ind_order, c in enumerate([c_v, c_a, c_j]):
        for ind_dim in range(c_p.shape[-1]):
            c[:, ind_dim] = 1 / dt**(ind_order + 1) * np.convolve(c_p[:, ind_dim].flatten(), A_c[ind_order + 1], mode='same')

    # Fix boundary conditions
    c_v[ 0, :] = c_v[1, :]

    c_a[ 0, :] = c_a[1, :]
    c_a[-1, :] = c_a[-2, :]

    c_j[ 0, :] = c_j[2, :]
    c_j[ 1, :] = c_j[2, :]
    c_j[-1, :] = c_j[-2, :]

    return c_v, c_a, c_j

def interpolate_b_spline(c_in, tau_in, t_range, t_res=0.005, n_d=0):
    """
    Calculate values for a B-spline with control points c_in and knots tau_in. The input control points are for the
    n_d'th derivative.
    """
    if len(t_range) == 2:
        t_valid = np.arange(t_range[0], t_range[1] + t_res, t_res)
    else:
        t_valid = t_range
    bspl = BSpline(tau_in, c_in, k=(K_CONST - n_d), extrapolate=False)
    s_out = bspl(t_valid)

    # Hand fixes for boundary conditions of edge functions
    # if n_d == 0:
    #     s_out[0, :] = c_in[1, :]

    return s_out, t_valid

def form_bspline(c_in, tau_in, k_in):
    """
    Forms a bspline for use within the trajectory function
    """
    bspl = BSpline(tau_in, c_in, k=k_in, extrapolate=False)

    return bspl