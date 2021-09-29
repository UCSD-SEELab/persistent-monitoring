import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    colors_fig = plt.get_cmap('tab10').colors

    theta = np.linspace(0, 2*np.pi, 40)
    b_rad = 10
    B_border = np.stack((b_rad * np.cos(theta), b_rad * np.sin(theta)), axis=-1)

    list_q = np.array([[ 0,  0],
                       [20, 20],
                       [44, 18],
                       [33,  -4]])
    plt.figure(1, figsize=(4, 3))

    plt.scatter(list_q[:, 0], list_q[:, 1], s=30, c='k', zorder=1)
    plt.scatter(list_q[:, 0], list_q[:, 1], s=10, c='white', zorder=2)

    for q in list_q:
        plt.plot(B_border[:, 0] + q[0], B_border[:, 1] + q[1], '-.', linewidth=2.5, color=colors_fig[0], zorder=3)

    list_q_plot = np.append(list_q, list_q[0:1, :], axis=0)
    plt.plot(list_q_plot[:, 0], list_q_plot[:, 1], 'k', zorder=0)
    plt.xlim([-13, 57])
    plt.ylim([-17, 33.5])

    plt.show()
    print(' ')