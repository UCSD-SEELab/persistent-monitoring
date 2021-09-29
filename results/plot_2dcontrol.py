import os
import sys
import time
import numpy as np
import pickle as pkl
import matplotlib.cm as colors
import matplotlib.pyplot as plt

from drones import *

if __name__ == '__main__':
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
            if not (file_target.lower().endswith('.pkl_updated')):
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

            # Create plots for each class of drone since each drone has different velocity and acceleration
            for drone_class in [Drone_Constant, Drone_Smith2012_Regions, Drone_Ostertag2019_Regions, Drone_Ostertag2020]:
                ind_drones_temp = [ind for ind, drone in enumerate(list_drones) if type(drone) == drone_class]
                list_drones_temp = [list_drones[ind] for ind in ind_drones_temp]
                num_drones_temp = len(ind_drones_temp)
                s_real_temp = s_real_all[ind_drones_temp, :, :]
                s_plan_temp = s_plan_all[ind_drones_temp, :, :, :]

                n_steps = s_plan_all.shape[-1]

                num_segs = 20
                seg_border = np.linspace(0, 12000, num_segs + 1).astype(int)

                plt.figure(1, figsize=(9.6, 7.2))
                # Plot the POIs in the environment
                arr_q = sim_env['list_q']
                plt.scatter(arr_q[:, 0], arr_q[:, 1], c='k', marker='o')

                h_planlines = []
                h_reallines = []
                list_legend = []
                for ind_plot, drone, s_real, s_plan in zip(range(num_drones_temp), list_drones_temp, s_real_temp, s_plan_temp):
                    ind_color = ind_plot % len(colors_fig)
                    h_temp = plt.plot(s_real[0, 0], s_real[1, 0], color=colors_fig[ind_color])
                    h_reallines.append(h_temp)

                    # h_temp = plt.plot(s_plan[0, 0, 0], s_plan[0, 1, 0], ':', color=colors_fig[ind_plot])
                    # h_planlines.append(h_temp)

                    list_legend += ['{1} v{2} a{3}'.format(ind_plot, type(drone).__name__, drone.vmax, drone.amax)]

                for ind_seg in range(num_segs):
                    for ind_plot, s_real, s_plan in zip(range(num_drones_temp), s_real_temp, s_plan_temp):
                        h_reallines[ind_plot][0].set_xdata(s_real[0, seg_border[ind_seg]:seg_border[ind_seg + 1]])
                        h_reallines[ind_plot][0].set_ydata(s_real[1, seg_border[ind_seg]:seg_border[ind_seg + 1]])

                        # h_planlines[ind_plot][0].set_xdata(s_plan[0, 0, seg_border[ind_seg]:seg_border[ind_seg + 1]])
                        # h_planlines[ind_plot][0].set_ydata(s_plan[0, 1, seg_border[ind_seg]:seg_border[ind_seg + 1]])

                    plt.legend(list_legend)
                    # Locate index of Nq80 and trim all following
                    file_base = os.path.splitext(file_target)[0]
                    ind_base = file_base.find('Nq')
                    file_base = file_base[:ind_base + 4]
                    file_save = '{0}_{1}_seg{2}.png'.format(file_base, drone_class.__name__, ind_seg)

                    plt.savefig(os.path.join(dir_img, file_save))

                plt.clf()







