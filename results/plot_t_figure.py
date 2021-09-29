import pickle as pkl
import os, sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(os.getcwd())

    filename_data = '20200310_224433.pkl'
    data = pkl.load(open(filename_data, 'rb'))
    t = [m_entry[:, -1] for m_entry in data['slack']]
    t_avg = np.array([np.average(t_entry) for t_entry in t]) * 1000
    t_std = np.array([np.std(t_entry) for t_entry in t]) * 1000
    m = np.array([m_entry[0, 0] for m_entry in data['slack']])

    plt.figure(1, figsize=(5,5))
    plt.plot(m, t_avg, linewidth=3)
    plt.xlabel('Number of Bases (m)', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.grid()

    filename_data = 'timing_gka.pkl'
    data_gka = pkl.load(open(filename_data, 'rb'))
    t_avg_gka = np.average(data_gka, axis=0)
    t_std_gka = np.std(data_gka, axis=0)

    plt.figure(2, figsize=(5, 5))
    plt.plot(np.arange(1, 8), t_avg_gka[1:-1], linewidth=3)
    plt.xlabel('Number Consecutive Observations', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.grid()

    plt.show()
    print(' ')