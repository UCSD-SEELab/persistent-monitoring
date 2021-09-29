import os

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_target = '20200305'
    list_files_Nq10 = ['20200305_133501_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133602_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133635_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133701_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133742_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133833_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_133928_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134014_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134049_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134130_Nq10_vmax20.0_amax30.0_B10.0.pkl']

    list_files_Nq20 = ['20200305_133936_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134041_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134222_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134314_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134357_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134533_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134615_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134707_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134745_Nq20_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_134830_Nq20_vmax20.0_amax30.0_B10.0.pkl',]

    list_files_Nq40 = ['20200305_135050_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_135228_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_135407_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_135810_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_140004_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_140127_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_140228_Nq40_vmax20.0_amax30.0_B10.0.pkl',
                        '20200305_140338_Nq40_vmax20.0_amax30.0_B10.0.pkl',]

    data_plot_Nq10 = np.zeros((len(list_files_Nq10), 4))
    data_plot_Nq20 = np.zeros((len(list_files_Nq20), 4))
    data_plot_Nq40 = np.zeros((len(list_files_Nq40), 4))

    for ind, file_Nq10 in enumerate(list_files_Nq10):
        data_in = pkl.load(open(os.path.join(dir_target, file_Nq10), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        if temp.shape[0] < 4:
            print('Error: Drone planner failed. Nq10 file {0}'.format(ind + 1))
            data_plot_Nq10[ind, :] = np.nan
            continue
        else:
            data_plot_Nq10[ind, :] = np.max(temp, axis=1)

    for ind, file_Nq20 in enumerate(list_files_Nq20):
        data_in = pkl.load(open(os.path.join(dir_target, file_Nq20), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        if temp.shape[0] < 4:
            print('Error: Drone planner failed. Nq20 file {0}'.format(ind + 1))
            data_plot_Nq10[ind, :] = np.nan
            continue
        else:
            data_plot_Nq20[ind, :] = np.max(temp, axis=1)

    for ind, file_Nq40 in enumerate(list_files_Nq40):
        data_in = pkl.load(open(os.path.join(dir_target, file_Nq40), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        if temp.shape[0] < 4:
            print('Error: Drone planner failed. Nq40 file {0}'.format(ind + 1))
            data_plot_Nq10[ind, :] = np.nan
            continue
        else:
            data_plot_Nq40[ind, :] = np.max(temp, axis=1)

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

    print(' ')
    place_ord = np.array([0, 1, 2, 3])
    x_off = 8
    list_config = ['o', 'o', 'o', 'o']
    plt.figure()
    for ind in range(4):
        ind_place = place_ord[ind]
        y_temp = np.array([data_Nq10_avg[ind_place], data_Nq20_avg[ind_place], data_Nq40_avg[ind_place]])
        x_temp = np.array([ind, ind + x_off, ind + 2*x_off])
        yerr_temp = np.array([data_Nq10_std[ind_place], data_Nq20_std[ind_place], data_Nq40_std[ind_place]])
        plt.errorbar(x=x_temp, y=y_temp, yerr=yerr_temp, fmt=list_config[ind])

    plt.xticks([1.5, 9.5, 17.5], labels=['10', '20', '40'])
    plt.legend(['Constant', 'First-Order', 'Ostertag2019', 'Proposed'], loc='upper right')

    plt.plot([-10, 30], [0, 0], 'k:')

    plt.ylabel('Max Covar (%)')
    plt.xlabel('# POI')
    plt.xlim([-1, 20])
    plt.show()

    print(' ')
