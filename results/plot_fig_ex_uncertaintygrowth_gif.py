import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':

    lambda_0 = np.array([25, 14, 12])
    eta = np.array([0.7, 0.3, 0.3]) / 6

    lim_x = [0, 130]
    lim_y = [10, 45]

    t1 = np.linspace(0, 30, 100).reshape([-1, 1])
    t2 = np.linspace(0, 60, 100).reshape([-1, 1])
    t3 = np.linspace(0, 120, 100).reshape([-1, 1])

    plt.figure(1, figsize=(4,6))
    ax_q1 = plt.subplot(311)
    ax_q2 = plt.subplot(312)
    ax_q4 = plt.subplot(313)

    lambda_t = t1 * eta + lambda_0
    ax_q1.plot(t1, lambda_t[:, 0], linewidth=3)
    ax_q1.set_xlim(lim_x)
    ax_q1.set_ylim(lim_y)
    ax_q1.set_xticklabels([])
    ax_q1.grid()

    ax_q2.plot(t1, lambda_t[:, 1], linewidth=3)
    ax_q2.set_xlim(lim_x)
    ax_q2.set_ylim(lim_y)
    ax_q2.set_xticklabels([])
    ax_q2.set_ylabel('Uncertainty', fontsize=14)
    ax_q2.grid()

    ax_q4.plot(t1, lambda_t[:, 2], linewidth=3)
    ax_q4.set_xlim(lim_x)
    ax_q4.set_ylim(lim_y)
    ax_q4.set_xlabel('Time (s)', fontsize=14)
    ax_q4.grid()

    plt.savefig('fig_covar_env_1.png', transparent=True)

    plt.figure(2, figsize=(4, 6))
    ax_q1 = plt.subplot(311)
    ax_q2 = plt.subplot(312)
    ax_q4 = plt.subplot(313)

    lambda_t = t2 * eta + lambda_0
    ax_q1.plot(t2, lambda_t[:, 0], linewidth=3)
    ax_q1.set_xlim(lim_x)
    ax_q1.set_ylim(lim_y)
    ax_q1.set_xticklabels([])
    ax_q1.grid()

    ax_q2.plot(t2, lambda_t[:, 1], linewidth=3)
    ax_q2.set_xlim(lim_x)
    ax_q2.set_ylim(lim_y)
    ax_q2.set_xticklabels([])
    ax_q2.set_ylabel('Uncertainty', fontsize=14)
    ax_q2.grid()

    ax_q4.plot(t2, lambda_t[:, 2], linewidth=3)
    ax_q4.set_xlim(lim_x)
    ax_q4.set_ylim(lim_y)
    ax_q4.set_xlabel('Time (s)', fontsize=14)
    ax_q4.grid()

    plt.savefig('fig_covar_env_2.png', transparent=True)

    plt.figure(3, figsize=(4, 6))
    ax_q1 = plt.subplot(311)
    ax_q2 = plt.subplot(312)
    ax_q4 = plt.subplot(313)

    lambda_t = t3 * eta + lambda_0
    ax_q1.plot(t3, lambda_t[:, 0], linewidth=3)
    ax_q1.set_xlim(lim_x)
    ax_q1.set_ylim(lim_y)
    ax_q1.set_xticklabels([])
    ax_q1.grid()

    ax_q2.plot(t3, lambda_t[:, 1], linewidth=3)
    ax_q2.set_xlim(lim_x)
    ax_q2.set_ylim(lim_y)
    ax_q2.set_xticklabels([])
    ax_q2.set_ylabel('Uncertainty', fontsize=14)
    ax_q2.grid()

    ax_q4.plot(t3, lambda_t[:, 2], linewidth=3)
    ax_q4.set_xlim(lim_x)
    ax_q4.set_ylim(lim_y)
    ax_q4.set_xlabel('Time (s)', fontsize=14)
    ax_q4.grid()

    plt.savefig('fig_covar_env_3.png', transparent=True)
    #plt.show()
    print(' ')