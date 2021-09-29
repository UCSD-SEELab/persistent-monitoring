import os

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_parent = '20200311'
    dir_files_Nq10 = os.path.join(dir_parent, 'Nq10_vmax20_amax30_B10')
    dir_files_Nq20 = os.path.join(dir_parent, 'Nq20_vmax20_amax30_B10')
    dir_files_Nq40 = os.path.join(dir_parent, 'Nq40_vmax20_amax30_B10')
    dir_files_Nq80 = os.path.join(dir_parent, 'Nq80_vmax20_amax30_B10')

    data_plot_Nq10 = np.zeros((0, 5))
    data_plot_Nq20 = np.zeros((0, 5))
    data_plot_Nq40 = np.zeros((0, 5))
    data_plot_Nq80 = np.zeros((0, 5))

    for file_Nq10 in os.listdir(dir_files_Nq10):
        data_in = pkl.load(open(os.path.join(dir_files_Nq10, file_Nq10), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        if data_plot_temp.size < 5:
            print('Error: {0}'.format(file_Nq10))
            continue

        data_plot_Nq10 = np.append(data_plot_Nq10, data_plot_temp, axis=0)

    for file_Nq20 in os.listdir(dir_files_Nq20):
        data_in = pkl.load(open(os.path.join(dir_files_Nq20, file_Nq20), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        if data_plot_temp.size < 5:
            print('Error: {0}'.format(file_Nq20))
            continue

        data_plot_Nq20 = np.append(data_plot_Nq20, data_plot_temp, axis=0)

    for file_Nq40 in os.listdir(dir_files_Nq40):
        data_in = pkl.load(open(os.path.join(dir_files_Nq40, file_Nq40), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        if data_plot_temp.size < 5:
            print('Error: {0}'.format(file_Nq40))
            continue

        data_plot_Nq40 = np.append(data_plot_Nq40, data_plot_temp, axis=0)

    for file_Nq80 in os.listdir(dir_files_Nq80):
        data_in = pkl.load(open(os.path.join(dir_files_Nq80, file_Nq80), 'rb'))
        results = data_in['results']['covar']

        covar_max = np.max(results[:, :, 1:], axis=2)
        covar_bound = data_in['drone']['drones'][-1].covar_bound + 0.5
        covar_end = covar_max[:, -2000:]
        covar_dronemax = np.max(covar_end, axis=1)
        data_plot_temp = np.append(covar_dronemax, covar_bound)
        data_plot_temp = data_plot_temp.reshape(1, -1)

        if data_plot_temp.size < 5:
            print('Error: {0}'.format(file_Nq80))
            continue

        data_plot_Nq80 = np.append(data_plot_Nq80, data_plot_temp, axis=0)

    data_Nq10_pct = np.zeros(data_plot_Nq10.shape)
    for ind, datum_plot_Nq10 in enumerate(data_plot_Nq10):
        data_Nq10_pct[ind, :] = 100 * (datum_plot_Nq10 / datum_plot_Nq10[-1] - 1)
    data_Nq10_std = np.std(data_Nq10_pct, axis=0)
    data_Nq10_avg = np.average(data_Nq10_pct, axis=0)

    data_Nq20_pct = np.zeros(data_plot_Nq20.shape)
    for ind, datum_plot_Nq20 in enumerate(data_plot_Nq20):
        data_Nq20_pct[ind, :] = 100 * (datum_plot_Nq20 / datum_plot_Nq20[-1] - 1)
    data_Nq20_std = np.std(data_Nq20_pct, axis=0)
    data_Nq20_avg = np.average(data_Nq20_pct, axis=0)

    data_Nq40_pct = np.zeros(data_plot_Nq40.shape)
    for ind, datum_plot_Nq40 in enumerate(data_plot_Nq40):
        data_Nq40_pct[ind, :] = 100 * (datum_plot_Nq40 / datum_plot_Nq40[-1] - 1)
    data_Nq40_std = np.std(data_Nq40_pct, axis=0)
    data_Nq40_avg = np.average(data_Nq40_pct, axis=0)

    data_Nq80_pct = np.zeros(data_plot_Nq80.shape)
    for ind, datum_plot_Nq80 in enumerate(data_plot_Nq80):
        data_Nq80_pct[ind, :] = 100 * (datum_plot_Nq80 / datum_plot_Nq80[-1] - 1)
    data_Nq80_std = np.std(data_Nq80_pct, axis=0)
    data_Nq80_avg = np.average(data_Nq80_pct, axis=0)

    print(' ')
    colors_fig = plt.get_cmap('tab10').colors
    x_off = 8
    plt.figure()
    for ind in range(4):
        y_temp = np.array([data_Nq10_avg[ind], data_Nq20_avg[ind], data_Nq40_avg[ind], data_Nq80_avg[ind]])
        x_temp = np.array([ind, ind + x_off, ind + 2*x_off, ind + 3*x_off])
        yerr_temp = np.array([data_Nq10_std[ind], data_Nq20_std[ind], data_Nq40_std[ind], data_Nq80_std[ind]])
        plt.errorbar(x=x_temp, y=y_temp, yerr=yerr_temp, fmt='o', color=colors_fig[ind])

    plt.xticks([1.5, 9.5, 17.5, 25.5], labels=['10', '20', '40', '80'])
    plt.legend(['Constant', 'First-Order', 'Ostertag2019', 'Proposed'], loc='upper right')

    plt.plot([-10, 30], [0, 0], 'k--', zorder=-1)

    plt.ylabel('Max Covar (%)', fontsize=14)
    plt.xlabel('Number of POIs', fontsize=14)
    plt.xlim([-1, 28])
    plt.grid()
    plt.show()

    print(' ')
