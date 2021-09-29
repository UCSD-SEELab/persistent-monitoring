import os

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_target = '20200305'
    list_files_2_0 = ['20200305_132659_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_132742_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_132833_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_132918_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_132959_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_133045_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_133138_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_133225_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_133332_Nq10_vmax10.0_amax30.0_B10.0.pkl',
                    '20200305_133427_Nq10_vmax10.0_amax30.0_B10.0.pkl']

    list_files_1_0 = ['20200305_133501_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133602_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133635_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133701_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133742_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133833_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_133928_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_134014_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_134049_Nq10_vmax20.0_amax30.0_B10.0.pkl',
                    '20200305_134130_Nq10_vmax20.0_amax30.0_B10.0.pkl']

    list_files_0_6 = ['20200305_134233_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134336_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134449_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134607_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134734_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134859_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_134955_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_135101_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_135215_Nq10_vmax30.0_amax30.0_B10.0.pkl',
                    '20200305_135316_Nq10_vmax30.0_amax30.0_B10.0.pkl']

    data_plot_2_0 = np.zeros((len(list_files_2_0), 4))
    data_plot_1_0 = np.zeros((len(list_files_1_0), 4))
    data_plot_0_6 = np.zeros((len(list_files_0_6), 4))

    for ind, file_2_0 in enumerate(list_files_2_0):
        data_in = pkl.load(open(os.path.join(dir_target, file_2_0), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        data_plot_2_0[ind, :] = np.max(temp, axis=1)

    for ind, file_1_0 in enumerate(list_files_1_0):
        data_in = pkl.load(open(os.path.join(dir_target, file_1_0), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        data_plot_1_0[ind, :] = np.max(temp, axis=1)

    for ind, file_0_6 in enumerate(list_files_0_6):
        data_in = pkl.load(open(os.path.join(dir_target, file_0_6), 'rb'))
        results = data_in['results']['covar']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        data_plot_0_6[ind, :] = np.max(temp, axis=1)

    data_2_0_pct = np.zeros(data_plot_2_0.shape)
    for ind, datum_plot_2_0 in enumerate(data_plot_2_0):
        data_2_0_pct[ind, :] = 100 * (datum_plot_2_0 / datum_plot_2_0[-1] - 1)
    data_2_0_std = np.std(data_2_0_pct, axis=0)
    data_2_0_avg = np.average(data_2_0_pct, axis=0)

    data_1_0_pct = np.zeros(data_plot_1_0.shape)
    for ind, datum_plot_1_0 in enumerate(data_plot_1_0):
        data_1_0_pct[ind, :] = 100 * (datum_plot_1_0 / datum_plot_1_0[-1] - 1)
    data_1_0_std = np.std(data_1_0_pct, axis=0)
    data_1_0_avg = np.average(data_1_0_pct, axis=0)

    data_0_6_pct = np.zeros(data_plot_0_6.shape)
    for ind, datum_plot_0_6 in enumerate(data_plot_0_6):
        data_0_6_pct[ind, :] = 100 * (datum_plot_0_6 / datum_plot_0_6[-1] - 1)
    data_0_6_std = np.std(data_0_6_pct, axis=0)
    data_0_6_avg = np.average(data_0_6_pct, axis=0)

    print(' ')
    place_ord = np.array([0, 1, 2, 3])
    x_off = 8
    list_config = ['o', 'o', 'o', 'o']
    plt.figure()
    for ind in range(4):
        ind_place = place_ord[ind]
        y_temp = np.array([data_2_0_avg[ind_place], data_1_0_avg[ind_place], data_0_6_avg[ind_place]])
        x_temp = np.array([ind, ind + x_off, ind + 2*x_off])
        yerr_temp = np.array([data_2_0_std[ind_place], data_1_0_std[ind_place], data_0_6_std[ind_place]])
        plt.errorbar(x=x_temp, y=y_temp, yerr=yerr_temp, fmt=list_config[ind])

    plt.xticks([1.5, 9.5, 17.5], labels=['2.0', '1.0', '0.66'])
    plt.legend(['Constant', 'First-Order', 'Ostertag2019', 'Proposed'], loc='upper right')

    plt.plot([-10, 30], [0, 0], 'k:')

    plt.ylabel('Max Covar (%)')
    plt.xlabel('Min. Samples Per Region')
    plt.xlim([-1, 20])
    plt.show()

    print(' ')
