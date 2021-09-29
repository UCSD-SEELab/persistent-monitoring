import os
import sys
import time
import numpy as np
import pickle as pkl
import matplotlib.cm as colors
import matplotlib.pyplot as plt

from drones import *

'''
    Drone_Constant
     0: 9 m/s
     1: 10 m/s
     2: 12 m/s
     3: 14 m/s
     4: 16 m/s
     5: 18 m/s

    Drone_Smith2012_Regions
     6: 9 m/s
     7: 10 m/s
     8: 12 m/s
     9: 14 m/s
    10: 16 m/s
    11: 18 m/s

    Drone_Ostertag2019_Regions
    12: 9 m/s
    13: 10 m/s
    14: 12 m/s
    15: 14 m/s
    16: 16 m/s
    17: 18 m/s

    Drone_Ostertag2020
    18: 12 m/s 16.3 m/s^2
    19: 12 m/s 17.3 m/s^2
    20: 12 m/s 18.2 m/s^2
    21: 12 m/s 19.2 m/s^2

    22: 14 m/s 16.3 m/s^2
    23: 14 m/s 17.3 m/s^2
    24: 14 m/s 18.2 m/s^2
    25: 14 m/s 19.2 m/s^2

    26: 16 m/s 16.3 m/s^2
    27: 16 m/s 17.3 m/s^2
    28: 16 m/s 18.2 m/s^2
    29: 16 m/s 19.2 m/s^2

    30: 18 m/s 16.3 m/s^2
    31: 18 m/s 17.3 m/s^2
    32: 18 m/s 18.2 m/s^2
    33: 18 m/s 19.2 m/s^2

    34: 20 m/s 16.3 m/s^2
    35: 20 m/s 17.3 m/s^2
    36: 20 m/s 18.2 m/s^2
    37: 20 m/s 19.2 m/s^2
'''

if __name__ == '__main__':
    colors_fig = plt.get_cmap('tab10').colors
    list_colors = [colors_fig[3], colors_fig[1], colors_fig[0], colors_fig[2]]

    dir_target = os.path.join(os.path.dirname(__file__), '..', 'results', '202101_tests')
    t_step = 0.1

    colors_fig = plt.get_cmap('tab10').colors
    list_colors = [colors_fig[3], colors_fig[1], colors_fig[0], colors_fig[2]]

    if not (os.path.isdir(dir_target)):
        print('ERROR: {0} does not exist.'.format(dir_target))
        sys.exit()

    for subdir_target in os.listdir(dir_target):
        dir_target_full = os.path.join(dir_target, subdir_target)
        if not(os.path.isdir(dir_target_full)):
            continue

        print('Processing files in {0}'.format(subdir_target))

        dir_img = os.path.join(dir_target_full, 'img')
        if not (os.path.isdir(dir_img)):
            print('Creating save directory for images')
            os.makedirs(dir_img)
        # else:
        #     print('Files already processed')
        #     continue

        for file_target in os.listdir(dir_target_full):
            if not (file_target.lower().endswith('.pkl')):
                continue

            print('Processing {0}'.format(file_target))

            try:
                data_in = pkl.load(open(os.path.join(dir_target_full, file_target), 'rb'))

                sim_env = data_in['env']
                list_drones = data_in['drone']['drones']
                s_plan_all = data_in['results']['s_plan']
                s_real_all = data_in['results']['s_real']
            except:
                print('ERROR: Failed loading pkl')
                continue

            # Create plots of single loop cycles for each target drone
            plt.figure(1, figsize=(7.6, 7.6))
            # Plot the POIs in the environment
            arr_q = sim_env['list_q']

            arr_rad = np.linspace(0, 2 * np.pi, 180)
            arr_rad = np.append(arr_rad, 2 * np.pi)
            x_circ = 10 * np.cos(arr_rad)
            y_circ = 10 * np.sin(arr_rad)

            for q in arr_q:
                temp_circle = plt.Circle((q[0], q[1]), 10, color=(0.8, 0.8, 0.8), zorder=-1)
                plt.gca().add_patch(temp_circle)

            plt.scatter(arr_q[:, 0], arr_q[:, 1], c='k', marker='o')


            list_targets = [2, 14, 27]
            for ind_target in list_targets:
                # Get drone plotting info
                drone = list_drones[ind_target]
                if type(drone).__name__ == 'Drone_Constant':
                    color_marker = colors_fig[3]
                    label = 'Constant'

                elif type(drone).__name__ == 'Drone_Smith2012_Regions':
                    color_marker = colors_fig[1]
                    label = 'Linear'

                elif type(drone).__name__ == 'Drone_Ostertag2019_Regions':
                    color_marker = colors_fig[5]
                    label = 'GKA 2019'

                elif (type(drone).__name__ == 'Drone_Ostertag2020') and (drone.amax == 17.3):
                    color_marker = colors_fig[2]
                    label = 'Proposed 90% accel'

                s_real = s_real_all[ind_target, :, :]
                s_plan = s_plan_all[ind_target, :, :, :]

                # Find indeces to map out single loops
                s_start = s_plan[0, :, 0]
                s_diff = s_plan[0, :, :] - s_start.reshape([-1, 1])
                ind_loopend = np.where(np.sqrt(np.sum(s_diff ** 2, axis=0)) < 10)[0]
                if not (ind_loopend.size > 1):
                    continue
                ind_loopend = np.array([val for ind, val in enumerate(ind_loopend[:-1]) if not (val == (ind_loopend[ind + 1] - 1))] + [ind_loopend[-1]])
                if not (ind_loopend.size > 1):
                    continue

                N_loops = ind_loopend.size - 1


                h_temp = plt.plot(s_real[0, 0], s_real[1, 0], c=color_marker, label=label)

                for ind_loop in [1]: #range(N_loops):
                    h_temp[0].set_xdata(s_real[0, ind_loopend[ind_loop]:ind_loopend[ind_loop + 1]])
                    h_temp[0].set_ydata(s_real[1, ind_loopend[ind_loop]:ind_loopend[ind_loop + 1]])

            plt.title('')
            plt.xlim([-10, 460])
            plt.ylim([-10, 460])

            # Locate index of Nq## and trim all following
            file_base = os.path.splitext(file_target)[0]
            ind_base = file_base.find('Nq')
            file_base = file_base[:ind_base + 4]
            file_save = '{0}_combo_loop{2}.png'.format(file_base, ind_target, ind_loop)
            file_save_zoom = '{0}_combo_loop{2}_zoom.png'.format(file_base, ind_target, ind_loop)
            file_save_legend = '{0}_combo_loop{2}_legend.png'.format(file_base, ind_target, ind_loop)

            plt.savefig(os.path.join(dir_img, file_save), dpi=300, transparent=True)
            plt.savefig(os.path.join(dir_img, file_save_zoom), dpi=900, transparent=True)

            plt.legend(fontsize=12)

            plt.savefig(os.path.join(dir_img, file_save_legend), dpi=300, transparent=True)

            plt.clf()







