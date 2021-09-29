import os
import sys
import time
import numpy as np
import pickle as pkl
import matplotlib.cm as colors
import matplotlib.pyplot as plt

from drones import *

def print_latex_table_line(list_args):
    for arg_in in list_args:
        print("{0:0.1f}\t& {1:0.1f}/{2:0.1f}\t& ".format(np.mean(arg_in), np.min(arg_in), np.max(arg_in)), end='')

    print(" ")

if __name__ == '__main__':
    dir_target = os.path.join(os.path.dirname(__file__), '..', 'results', '202101_tests')
    t_step = 0.1

    colors_fig = plt.get_cmap('tab10').colors
    list_colors = [colors_fig[3], colors_fig[1], colors_fig[0], colors_fig[2]]

    if not (os.path.isdir(dir_target)):
        print('ERROR: {0} does not exist.'.format(dir_target))
        sys.exit()


    list_results = []

    for subdir_target in os.listdir(dir_target):
        dir_target_full = os.path.join(dir_target, subdir_target)
        if not(os.path.isdir(dir_target_full)):
            continue

        print('Processing files in {0}'.format(subdir_target))

        dir_img = os.path.join(dir_target_full, 'fig')
        if not (os.path.isdir(dir_img)):
            print('Creating save directory for images')
            os.makedirs(dir_img)
        # else:
        #     print('Files already processed')
        #     continue

        arr_zerr = np.zeros((0,0))
        arr_xyerr = np.zeros((0,0))
        arr_maxuncert = np.zeros((0,0))
        arr_maxvelacc = np.zeros((0,0))
        arr_num_obs = np.zeros((0,0))
        arr_num_miss = np.zeros((0,0))
        arr_avg_tloop = np.zeros((0,0))

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
                b_samp = data_in['results']['b_samp']
                covar = data_in['results']['covar']
                time_info = data_in['time']
            except:
                print('ERROR: Failed loading pkl')
                continue

            num_drones = s_plan_all.shape[0]
            num_q = covar.shape[-1] - 1

            # Convert b_samp to indeces of covar that have samples
            list_b_samp = ((covar[0, 1:, 0] - covar[0, :-1, 0]) < (time_info['t_step'] / 10)).tolist()
            b_samp_pre = np.array(list_b_samp + [False])
            b_samp_post = np.array([False] + list_b_samp)


            # Process results for all drones, keeping order since each drone has different velocity and acceleration
            temp_zerr = s_real_all[:, 2, :] - s_plan_all[:, 0, 2, :]

            max_zerr = np.max(np.abs(temp_zerr), axis=1).reshape(-1, 1)

            if arr_zerr.size == 0:
                arr_zerr = max_zerr
            else:
                arr_zerr = np.append(arr_zerr, max_zerr, axis=1)

            # Calculte magnitude XY error
            temp_xyerr = (s_real_all[:, :2, :] - s_plan_all[:, 0, :2, :])**2
            max_xyerr = np.max(np.sqrt(np.sum(temp_xyerr, axis=1)), axis=1).reshape(-1, 1)
            avg_xyerr = np.mean(np.sqrt(np.sum(temp_xyerr, axis=1)), axis=1).reshape(-1, 1)

            # Calculate planned loop time and pct of points measured
            avg_tloop = np.zeros((num_drones))
            list_pct_obs = [None] * num_drones
            arr_num_obs_temp = np.zeros((num_drones, 2))
            arr_num_miss_temp = np.zeros((num_drones, 2))
            covar_diff = np.append(covar[:, b_samp_post, :1], covar[:, b_samp_post, 1:] - covar[:, b_samp_pre, 1:], axis=2)

            for ind_drone, s_start, s_plan in zip(range(num_drones), s_plan_all[:, 0, :, :1], s_plan_all[:, 0, :, :]):
                s_diff = s_plan - s_start
                ind_loopend = np.where(np.sqrt(np.sum(s_diff**2, axis=0)) < 1)[0]
                if not(ind_loopend.size > 1):
                    continue
                ind_loopend = np.array([val for ind, val in enumerate(ind_loopend[:-1]) if not(val == (ind_loopend[ind + 1] - 1))] + [ind_loopend[-1]])
                if not (ind_loopend.size > 1):
                    continue
                avg_tloop[ind_drone] = np.average(ind_loopend[1:] - ind_loopend[:-1]) * time_info['t_step']

                num_loops = ind_loopend.size - 1
                arr_obstaken = np.zeros((num_loops, num_q))
                for obstaken, ind_loop_start, ind_loop_end in zip(arr_obstaken, ind_loopend[:-1], ind_loopend[1:]):
                    t_start = (ind_loop_start - 0.1) * time_info['t_step']
                    t_end = (ind_loop_end + 0.1) * time_info['t_step']
                    covar_diff_seg = covar_diff[ind_drone, np.logical_and(covar_diff[ind_drone, :, 0] > t_start, covar_diff[ind_drone, :, 0] < t_end), :]
                    obstaken[:] = np.sum(covar_diff_seg < 0, axis=0)[1:]
                list_pct_obs[ind_drone] = arr_obstaken
                arr_num_obs_temp[ind_drone, :] = np.sum(arr_obstaken[-2:, :] >= 1, axis=1)
                arr_num_miss_temp[ind_drone, :] = np.sum(arr_obstaken[-2:, :] < 1, axis=1)

            if arr_num_obs.size == 0:
                arr_num_obs = arr_num_obs_temp
                arr_num_miss = arr_num_miss_temp
            else:
                arr_num_obs = np.append(arr_num_obs, arr_num_obs_temp, axis=1)
                arr_num_miss = np.append(arr_num_miss, arr_num_miss_temp, axis=1)

            if arr_xyerr.size == 0:
                arr_xyerr = max_xyerr
            else:
                arr_xyerr = np.append(arr_xyerr, max_xyerr, axis=1)

            if arr_avg_tloop.size == 0:
                arr_avg_tloop = avg_tloop.reshape(-1, 1)
            else:
                arr_avg_tloop = np.append(arr_avg_tloop, avg_tloop.reshape(-1, 1), axis=1)

            covar_bound = list_drones[-1].covar_bound
            max_covar = 100 * (np.max(np.max(covar[:, covar.shape[1] // 2:, 1:], axis=1), axis=1).reshape(-1, 1) - covar_bound) / covar_bound
            if arr_maxuncert.size == 0:
                arr_maxuncert = max_covar
            else:
                arr_maxuncert = np.append(arr_maxuncert, max_covar, axis=1)

        arr_pct_meas = arr_num_obs / (arr_num_obs + arr_num_miss)

        # Plot collected results, assume list_drones is the same for all test conditions in the folder
        """
        # Generate plot for XY and Z error vs max uncertainty
        plt.figure(1, figsize=(14.4, 7.2))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        ind_color_constant = 0
        ind_color_smith = 0
        ind_color_ostertag2019 = 0
        ind_color_ostertag2020 = 0

        '''
        Drone_Constant
        2: 10 m/s
        3: 12 m/s
        4: 16 m/s
        
        Drone_Smith2012_Regions
        7: 10 m/s
        8: 12 m/s
        9: 16 m/s
        
        Drone_Ostertag2019_Regions
        12: 10 m/s
        13: 12 m/s
        14: 16 m/s
        
        Drone_Ostertag2020
        18: 16 m/s 17.3 m/s^2
        20: 16 m/s 18.2 m/s^2
        21: 20 m/s 18.2 m/s^2
        22: 16 m/s 19.2 m/s^2
        23: 20 m/s 19.2 m/s^2
        '''

        ind_drones_valid = [2, 3, 4, 7, 8, 9, 12, 13, 14, 20, 21, 22, 23]
        list_legend = ['Constant 10 m/s',
                       'Constant 12 m/s',
                       'Constant 16 m/s',
                       'Linear 10 m/s',
                       'Linear 12 m/s',
                       'Linear 16 m/s',
                       'GKA 2019 10 m/s',
                       'GKA 2019 12 m/s',
                       'GKA 2019 16 m/s',
                       'Proposed 16 m/s 18.2 m/s^2',
                       'Proposed 20 m/s 18.2 m/s^2',
                       'Proposed 16 m/s 19.2 m/s^2',
                       'Proposed 20 m/s 19.2 m/s^2']

        for ind_drone, drone, zerr, xyerr, max_covar in zip(range(len(list_drones)), list_drones, arr_zerr, arr_xyerr, arr_maxuncert):
            # Select which elements to plot to help relieve clutter
            if not(ind_drone in ind_drones_valid):
                continue

            if type(drone).__name__ == 'Drone_Constant':
                char_marker = 'x'
                color_marker = colors_fig[ind_color_constant]
                ind_color_constant += 1
            elif type(drone).__name__ == 'Drone_Smith2012_Regions':
                char_marker = '+'
                color_marker = colors_fig[ind_color_smith]
                ind_color_smith += 1
            elif type(drone).__name__ == 'Drone_Ostertag2019_Regions':
                char_marker = 'v'
                color_marker = colors_fig[ind_color_ostertag2019]
                ind_color_ostertag2019 += 1
            elif type(drone).__name__ == 'Drone_Ostertag2020':
                char_marker = 'o'
                color_marker = colors_fig[ind_color_ostertag2020]
                ind_color_ostertag2020 += 1

            ax1.scatter(xyerr, max_covar, marker=char_marker, color=color_marker)
            ax2.scatter(zerr, max_covar, marker=char_marker, color=color_marker)

        # ax1.set_xscale('log')
        # ax2.set_xscale('log')
        ax1.set_xlim([0, 20])
        ax1.set_ylim([-25, 200])
        ax2.set_xlim([0, 20])
        ax2.set_ylim([-25, 200])

        ax1.set_title(subdir_target)
        ax1.set_xlabel('XY Error (m)')
        ax1.set_ylabel('Normalized Max Uncertainty (pct)')
        ax2.set_xlabel('Z Error (m)')
        ax2.set_ylabel('Normalized Max Uncertainty (pct)')

        ax2.legend(list_legend)

        file_save = '{0}_maxuncert_v_xyerr_close.png'.format(subdir_target)
        plt.savefig(os.path.join(dir_img, file_save))
        plt.clf()
        """

        # Plot pct_measured
        plt.figure(2, figsize=(14.4, 7.2))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        list_legend = ['Constant',
                       'Linear',
                       'GKA 2019',
                       'Proposed 95% accel',
                       'Proposed 100% accel']
        list_b_legend = np.ones(len(list_legend))
        for ind_drone, drone, pct_meas, xyerr, max_covar, avg_tloop in zip(range(len(list_drones)), list_drones, arr_pct_meas, arr_xyerr, arr_maxuncert, arr_avg_tloop):
            if type(drone).__name__ == 'Drone_Constant':
                char_marker = 'x'
                color_marker = colors_fig[3]
                if list_b_legend[0]:
                    label = 'Constant'
                    xoff = -0.2
                    list_b_legend[0] = 0
                else:
                    label = '_nolegend_'

                print("Constant \t& {0:0.1f}\t& --\t& ".format(drone.vmax), end = '')

            elif type(drone).__name__ == 'Drone_Smith2012_Regions':
                char_marker = '+'
                color_marker = colors_fig[1]
                if list_b_legend[1]:
                    label = 'Linear'
                    xoff = 0.0
                    list_b_legend[1] = 0
                else:
                    label = '_nolegend_'

                print("Linear \t& {0:0.1f}\t& --\t& ".format(drone.vmax), end = '')

            elif type(drone).__name__ == 'Drone_Ostertag2019_Regions':
                char_marker = 'v'
                color_marker = colors_fig[5]
                if list_b_legend[2]:
                    label = 'GKA 2019'
                    xoff = 0.2
                    list_b_legend[2] = 0
                else:
                    label = '_nolegend_'

                print("GKA 2019 \t& {0:0.1f}\t& --\t& ".format(drone.vmax), end = '')

            elif (type(drone).__name__ == 'Drone_Ostertag2020') and (drone.amax == 18.2):
                char_marker = 'o'
                color_marker = colors_fig[2]
                if list_b_legend[3]:
                    label = 'Proposed 95% accel'
                    xoff = 0.0
                    list_b_legend[3] = 0
                else:
                    label = '_nolegend_'

                print("Prosposed\t& {0:0.1f}\t& {1:0.1f}\t& ".format(drone.vmax, drone.amax), end = '')

            elif (type(drone).__name__ == 'Drone_Ostertag2020') and (drone.amax == 19.2):
                char_marker = 'o'
                color_marker = colors_fig[8]
                if list_b_legend[4]:
                    label = 'Proposed 100% accel'
                    xoff = 0.2
                    list_b_legend[4] = 0
                else:
                    label = '_nolegend_'

                print("Prosposed\t& {0:0.1f}\t& {1:0.1f}\t& ".format(drone.vmax, drone.amax), end = '')

            else:
                continue

            print_latex_table_line([max_covar, xyerr, pct_meas*100, avg_tloop])

            print("PCT Measured")
            print("  avg={0}".format(np.mean(pct_meas)))
            print("  min={0}".format(np.min(pct_meas)))
            print("  max={0}".format(np.max(pct_meas)))
            print(" ")
            print("XY Error")
            print("  avg={0}".format(np.mean(xyerr)))
            print("  min={0}".format(np.min(xyerr)))
            print("  max={0}".format(np.max(xyerr)))
            print(" ")
            print("Max Covar")
            print("  avg={0}".format(np.mean(max_covar)))
            print("  min={0}".format(np.min(max_covar)))
            print("  max={0}".format(np.max(max_covar)))
            print(" ")
            print("t_loop")
            print("  avg={0}".format(np.mean(avg_tloop)))
            print("  min={0}".format(np.min(avg_tloop)))
            print("  max={0}".format(np.max(avg_tloop)))
            print(" ")

            pct_plot = (pct_meas[::2] + pct_meas[1::2]) / 2

            ax1.scatter(drone.vmax * np.ones(pct_plot.shape) + xoff / 10, pct_plot, marker=char_marker, color=color_marker, label=label)
            ax2.scatter(avg_tloop + xoff * 10, pct_plot, marker=char_marker, color=color_marker, label=label)

        # ax1.set_xscale('log')
        # ax2.set_xscale('log')
        # ax1.set_xlim([0, 20])
        # ax1.set_ylim([0.75, 1.05])
        # ax2.set_xlim([0, 20])
        # ax2.set_ylim([0.75, 1.05])

        ax1.set_title(subdir_target)
        ax1.set_xlabel('Max Velocity (m/s)')
        ax1.set_ylabel('Pct Measured per Loop')
        ax2.set_xlabel('Avg Loop Time (s)')
        ax2.set_ylabel('Pct Measured per Loop')

        ax2.legend(list_legend)

        file_save = '{0}_pct_meas2.png'.format(subdir_target)
        plt.savefig(os.path.join(dir_img, file_save))
        plt.clf()











