import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    list_files_vmax25 = ['20200227_133654_Nq30_vmax25_B10.pkl',
                         '20200227_133914_Nq30_vmax25_B10.pkl',
                         '20200227_134207_Nq30_vmax25_B10.pkl']

    list_files_vmax10 = ['20200227_134905_Nq30_vmax10_B10.pkl',
                         '20200227_135221_Nq30_vmax10_B10.pkl',
                         '20200227_140213_Nq30_vmax10_B10.pkl']


    data_plot_25 = np.zeros((len(list_files_vmax25), 4))
    data_plot_10 = np.zeros((len(list_files_vmax10), 4))

    for ind, file_vmax25 in enumerate(list_files_vmax25):
        data_in = pkl.load(open(file_vmax25, 'rb'))
        results = data_in['results']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        data_plot_25[ind, :] = np.max(temp, axis=1)

    for ind, file_vmax10 in enumerate(list_files_vmax10):
        data_in = pkl.load(open(file_vmax10, 'rb'))
        results = data_in['results']

        test = results[0, :, 1:]
        covar_max = np.max(results[:, :, 1:], axis=2)
        temp = covar_max[:, -2000:]

        data_plot_10[ind, :] = np.max(temp, axis=1)

    data_10_pct = np.zeros(data_plot_10.shape)
    for ind, datum_plot_10 in enumerate(data_plot_10):
        data_10_pct[ind, :] = 100 * (datum_plot_10 / datum_plot_10[1] - 1)
    data_10_std = np.std(data_10_pct, axis=0)
    data_10_avg = np.average(data_10_pct, axis=0)

    data_25_pct = np.zeros(data_plot_25.shape)
    for ind, datum_plot_25 in enumerate(data_plot_25):
        data_25_pct[ind, :] = 100 * (datum_plot_25 / datum_plot_25[1] - 1)
    data_25_std = np.std(data_25_pct, axis=0)
    data_25_avg = np.average(data_25_pct, axis=0)

    print(' ')
    place_ord = np.array([0, 3, 2, 1])
    x_off = 8
    list_config = ['o', 'or', 'og', 'ob']
    plt.figure()
    for ind in range(4):
        ind_place = place_ord[ind]
        y_temp = np.array([data_10_avg[ind_place], data_25_avg[ind_place]])
        x_temp = np.array([ind, ind + x_off])
        yerr_temp = np.array([data_10_std[ind_place], data_25_std[ind_place]])
        plt.errorbar(x=x_temp, y=y_temp, yerr=yerr_temp, fmt=list_config[ind])

    plt.xticks([1.5, 9.5], labels=['10 m/s', '25 m/s'])
    plt.legend(['Constant', 'First-Order', 'Ostertag2019', 'Proposed'], loc='upper left')
    plt.ylabel('Max Covar (%)')
    plt.show()

    print(' ')
