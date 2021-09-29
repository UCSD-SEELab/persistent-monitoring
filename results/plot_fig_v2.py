import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_parent = '20200311'
    dir_files_vmax10 = os.path.join(dir_parent, 'Nq40_vmax10_amax30_B10')
    dir_files_vmax20 = os.path.join(dir_parent, 'Nq40_vmax20_amax30_B10')
    dir_files_vmax30 = os.path.join(dir_parent, 'Nq40_vmax30_amax30_B10')

    data_plot_10 = np.zeros((0, 5))
    for file in os.listdir(dir_files_vmax10):
        if not (file.lower().endswith('.pkl')):
            continue

        data_in = pkl.load(open(os.path.join(dir_files_vmax10, file), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        data_plot_10 = np.append(data_plot_10, data_plot_temp, axis=0)

    data_plot_20 = np.zeros((0, 5))
    for file in os.listdir(dir_files_vmax20):
        if not (file.lower().endswith('.pkl')):
            continue

        data_in = pkl.load(open(os.path.join(dir_files_vmax20, file), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        """
        plt.figure(1)
        for ind_drone in range(4):
            plt.subplot(411 + ind_drone)
            test = results[ind_drone, :, :]
            for ind_col in range(1, test.shape[1]):
                plt.plot(test[:, 0], test[:, ind_col])
            plt.plot(test[:, 0], covar_max[ind_drone, :], 'k', linewidth=3)
        plt.figure(2)
        for ind_drone in range(4):
            plt.plot(test[:, 0], covar_max[ind_drone, :] / covar_bound,  linewidth=3)
        plt.show()
        """
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        data_plot_20 = np.append(data_plot_20, data_plot_temp, axis=0)

    data_plot_30 = np.zeros((0, 5))
    for file in os.listdir(dir_files_vmax30):
        if not (file.lower().endswith('.pkl')):
            continue

        data_in = pkl.load(open(os.path.join(dir_files_vmax30, file), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        if data_plot_temp.size < 5:
            print('Error: {0}'.format(file))
            continue

        data_plot_30 = np.append(data_plot_30, data_plot_temp, axis=0)

    data_10_pct = np.zeros(data_plot_10.shape)
    for ind_row, datum_plot_10 in enumerate(data_plot_10):
        data_10_pct[ind_row, :] = 100 * (datum_plot_10 / datum_plot_10[-1] - 1)
    data_10_std = np.std(data_10_pct, axis=0)
    data_10_avg = np.average(data_10_pct, axis=0)

    data_20_pct = np.zeros(data_plot_20.shape)
    for ind_row, datum_plot_20 in enumerate(data_plot_20):
        data_20_pct[ind_row, :] = 100 * (datum_plot_20 / datum_plot_20[-1] - 1)
    data_20_std = np.std(data_20_pct, axis=0)
    data_20_avg = np.average(data_20_pct, axis=0)

    data_30_pct = np.zeros(data_plot_30.shape)
    for ind_row, datum_plot_30 in enumerate(data_plot_30):
        data_30_pct[ind_row, :] = 100 * (datum_plot_30 / datum_plot_30[-1] - 1)
    data_30_std = np.std(data_30_pct, axis=0)
    data_30_avg = np.average(data_30_pct, axis=0)

    x_off = 8
    colors_fig = plt.get_cmap('tab10').colors
    plt.figure()
    for ind in range(4):
        y_temp = np.array([data_10_avg[ind], data_20_avg[ind], data_30_avg[ind]])
        x_temp = np.array([ind, ind + x_off, ind + 2*x_off])
        yerr_temp = np.array([data_10_std[ind], data_20_std[ind], data_30_std[ind]])
        plt.errorbar(x=x_temp, y=y_temp, yerr=yerr_temp, fmt='o', color=colors_fig[ind], zorder=1)

    plt.xticks([1.5, 9.5, 17.5], labels=['10 m/s', '20 m/s', '30 m/s'])
    plt.legend(['Constant', 'First-Order', 'Ostertag 2019 (Direct)', 'Ostertag 2020 (Feasible)'], loc='lower left')
    plt.plot([-10, 25], [0, 0], 'k--', zorder=-1)
    plt.ylabel('Max Covar (%)', fontsize=14)
    plt.xlabel('Max Velocity (m/s)', fontsize=14)
    plt.xlim([-2, 22])
    plt.ylim([-45, 20])
    plt.grid()
    plt.show()

    print(' ')
