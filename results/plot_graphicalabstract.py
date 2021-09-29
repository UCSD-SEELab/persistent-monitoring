import os
import numpy as np
import pickle as pkl

from env_models import Model_Fig1

import matplotlib.cm as colors
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dir_target = os.path.join('..', 'results', '20200309')
    dir_target = os.path.join('..', 'results', '20200310')
    # file_target = '20200309_173917_Nq20_vmax15.0_amax10.0_B10.0.pkl'
    file_target_long = '20200310_020458_Nq20_vmax15.0_amax10.0_B10.0.pkl'

    colors_fig = plt.get_cmap('tab10').colors

    data_in = pkl.load(open(os.path.join(dir_target, file_target_long), 'rb'))

    sim_env = data_in['env']
    list_drones = data_in['drone']['drones']

    # Background image
    #plt.figure(1, figsize=[8, 6])#, dpi=300)
    plt.figure(1, figsize=np.array([8, 6])*5.5/8)
    plt.scatter(sim_env['list_q'][:, 0], sim_env['list_q'][:, 1], s=30, c='k', zorder=1)
    plt.scatter(sim_env['list_q'][:, 0], sim_env['list_q'][:, 1], s=10, c='white', zorder=2)
    list_q = list_drones[0].list_q
    list_q_plot = np.append(list_q, list_q[0:1, :], axis=0)
    ind_q1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]
    ind_q2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 11]
    list_q_1 = list_q[ind_q1]
    list_q_2 = list_q[ind_q2]

    plt.plot(list_q_1[:, 0], list_q_1[:, 1], 'k--', zorder=0)
    plt.plot(list_q_2[:, 0], list_q_2[:, 1], 'k--', zorder=0)
    plt.xticks([], [])
    plt.yticks([], [])


    s_p_direct = data_in['results']['s'][2, 0, :, :]

    poi1 = np.array([73.2, 86.2])
    theta = np.linspace(0, 2*np.pi, 40)
    B_border = np.stack((13 * np.cos(theta), 13 * np.sin(theta)), axis=-1) + poi1
    plt.plot(B_border[:, 0], B_border[:, 1], '-.', linewidth=2.5, color=colors_fig[0])

    poi2 = np.array([122, 102.5])
    B_border = np.stack((13 * np.cos(theta), 13 * np.sin(theta)), axis=-1) + poi2
    plt.plot(B_border[:, 0], B_border[:, 1], '-.', linewidth=2.5, color=colors_fig[0])

    plt.savefig('fig_ga_base_transparent.png', transparent=True)

    # Foreground mid row

    s_p_first = data_in['results']['s'][0, 0, :, :]
    s_p_direct = data_in['results']['s'][2, 0, :, :]
    s_p_feasible = data_in['results']['s'][3, 0, :, :]

    poi1 = sim_env['list_q'][1, :]
    theta = np.linspace(0, 2*np.pi, 40)
    B_border = np.stack((10 * np.cos(theta), 10 * np.sin(theta)), axis=-1) + poi1

    plt.plot([5, 5, 27, 27, 5], [123, 145, 145, 123, 123], color=(0.2, 0.2, 0.2))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    #plt.savefig('fig_ga_base_transparent.png', transparent=True)

    plt.figure(2, figsize=[8, 2.5])
    ax_first = plt.subplot(131)
    ax_first.scatter(poi1[0], poi1[1], s=40, c='k', zorder=1)
    ax_first.scatter(poi1[0], poi1[1], s=15, c='white', zorder=2)
    ax_first.plot(list_q_plot[:, 0], list_q_plot[:, 1], 'k--', zorder=0)
    ax_first.plot(B_border[:, 0], B_border[:, 1], 'k-.', zorder=-1)
    ax_first.plot(s_p_direct[0, :], s_p_direct[1, :], color=colors_fig[1], zorder=0)
    ax_first.set_xlim([5, 27])
    ax_first.set_ylim([123, 145])
    ax_first.set_xticks([], [])
    ax_first.set_yticks([], [])
    ax_first.set_title('First-Order', color=colors_fig[1])

    ax_direct = plt.subplot(132)
    ax_direct.scatter(poi1[0], poi1[1], s=40, c='k', zorder=1)
    ax_direct.scatter(poi1[0], poi1[1], s=15, c='white', zorder=2)
    ax_direct.plot(list_q_plot[:, 0], list_q_plot[:, 1], 'k--', zorder=0)
    ax_direct.plot(B_border[:, 0], B_border[:, 1], 'k-.', zorder=-1)
    ax_direct.plot(s_p_direct[0, :], s_p_direct[1, :], color=colors_fig[2], zorder=0)
    ax_direct.set_xlim([5, 27])
    ax_direct.set_ylim([123, 145])
    ax_direct.set_xticks([], [])
    ax_direct.set_yticks([], [])
    ax_direct.set_title('Direct Planner', color=colors_fig[2])

    ax_feasible = plt.subplot(133)
    ax_feasible.scatter(poi1[0], poi1[1], s=40, c='k', zorder=1)
    ax_feasible.scatter(poi1[0], poi1[1], s=15, c='white', zorder=2)
    ax_feasible.plot(list_q_plot[:, 0], list_q_plot[:, 1], 'k--', zorder=0)
    ax_feasible.plot(B_border[:, 0], B_border[:, 1], 'k-.', zorder=-1)
    ax_feasible.plot(s_p_feasible[0, :], s_p_feasible[1, :], color=colors_fig[3], zorder=0)
    ax_feasible.set_xlim([5, 27])
    ax_feasible.set_ylim([123, 145])
    ax_feasible.set_xticks([], [])
    ax_feasible.set_yticks([], [])
    ax_feasible.set_title('Feasible Planner', color=colors_fig[3])

    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08)
    plt.savefig('fig_ga_detail_transparent.png', transparent=True)

    plt.figure(3, figsize=[8, 3])
    s_v_first = data_in['results']['s'][0, 1, :, :]
    s_v_direct = data_in['results']['s'][2, 1, :, :]
    s_v_feasible = data_in['results']['s'][3, 1, :, :]

    s_a_first = data_in['results']['s'][0, 2, :, :]
    s_a_direct = data_in['results']['s'][2, 2, :, :]
    s_a_feasible = data_in['results']['s'][3, 2, :, :]

    b_arr_valid_first = np.sqrt(np.sum(np.power(s_p_first.T - poi1, 2), axis=1)) < 10
    b_arr_valid_direct = np.sqrt(np.sum(np.power(s_p_direct.T - poi1, 2), axis=1)) < 10
    b_arr_valid_feasible = np.sqrt(np.sum(np.power(s_p_feasible.T - poi1, 2), axis=1)) < 10

    ind_valid_first = np.array([11713, 11725])
    # ind_valid_first = np.array([420, 433])
    ind_range_first = np.arange(ind_valid_first[1] + 13, ind_valid_first[0] - 14, -1)
    ind_valid_direct = np.array([11661, 11690])
    # ind_valid_direct = np.array([420, 449])
    ind_range_direct = np.arange(ind_valid_direct[1] + 29, ind_valid_direct[0] - 30, -1)
    ind_valid_feasible = np.array([11361, 11410])
    # ind_valid_feasible = np.array([653, 702])
    ind_range_feasible = np.arange(ind_valid_feasible[1] + 49, ind_valid_feasible[0] - 50, -1)

    s_v_mag_first = np.sqrt(np.sum(np.power(s_v_first, 2), axis=0))[ind_range_first]
    s_v_mag_direct = np.sqrt(np.sum(np.power(s_v_direct, 2), axis=0))[ind_range_direct]
    s_v_mag_feasible = np.sqrt(np.sum(np.power(s_v_feasible, 2), axis=0))[ind_range_feasible]

    s_a_mag_first = np.sqrt(np.sum(np.power(s_a_first, 2), axis=0))[ind_range_first]
    s_a_mag_direct = np.sqrt(np.sum(np.power(s_a_direct, 2), axis=0))[ind_range_direct]
    s_a_mag_feasible = np.sqrt(np.sum(np.power(s_a_feasible, 2), axis=0))[ind_range_feasible]

    t_step = 0.1
    t_first = t_step * (ind_range_first[::-1] - ind_valid_first[0])
    t_direct = t_step * (ind_range_direct[::-1] - ind_valid_direct[0])
    t_feasible = t_step * (ind_range_feasible[::-1] - ind_valid_feasible[0])

    t_first_10pct = (ind_valid_first[1] - ind_valid_first[0]) * t_step / 10
    t_direct_10pct = (ind_valid_direct[1] - ind_valid_direct[0]) * t_step / 10
    t_feasible_10pct = (ind_valid_feasible[1] - ind_valid_feasible[0]) * t_step / 10

    t_first_end = (ind_valid_first[1] - ind_valid_first[0]) * t_step
    t_direct_end = (ind_valid_direct[1] - ind_valid_direct[0]) * t_step
    t_feasible_end = (ind_valid_feasible[1] - ind_valid_feasible[0]) * t_step

    # Covar info
    ind_cv_first = np.arange(12869, 12913)
    ind_cv_direct = np.arange(12793, 12891)
    ind_cv_feasible = np.arange(12440, 12607)
    covar = data_in['results']['covar']
    cv_first = covar[1][ind_cv_first, -1]
    t_cv_first = covar[1][ind_cv_first, 0] - 1171.3
    cv_direct = covar[2][ind_cv_direct, -1]
    t_cv_direct = covar[2][ind_cv_direct, 0] - 1166.5
    cv_feasible = covar[3][ind_cv_feasible, -1]
    t_cv_feasible = covar[3][ind_cv_feasible, 0] - 1136.4

    v_range = [-0.75, 16.5]
    a_range = [-0.5, 11]
    a_end_buffer = 0.15

    # First-order Plan
    ax_first_v = plt.subplot(331)
    ax_first_v.plot(t_first, s_v_mag_first, color=colors_fig[1])
    ax_first_v.set_xlim([0 - t_first_10pct, t_first_end + t_first_10pct + a_end_buffer])
    ax_first_v.set_ylim(v_range)
    ax_first_v.set_xticks([], [])
    ax_first_v.set_ylabel('Velocity')

    ax_first_a = plt.subplot(334)
    ax_first_a.plot(t_first, s_a_mag_first, color=colors_fig[1])
    ax_first_a.plot([0.65, 0.65], [0, 100], color=colors_fig[1])
    ax_first_a.set_xlim([0 - t_first_10pct, t_first_end + t_first_10pct + a_end_buffer])
    ax_first_a.set_ylim(a_range)
    ax_first_a.set_xticks([], [])
    ax_first_a.set_ylabel('Acceleration')

    ax_first_lambda = plt.subplot(337)
    ax_first_lambda.plot(t_cv_first, cv_first, color=colors_fig[1])
    ax_first_lambda.set_xlim([0 - t_first_10pct, t_first_end + t_first_10pct + a_end_buffer])
    ax_first_lambda.set_ylim([-0.75, 65])
    ax_first_lambda.set_ylabel('Uncertainty')

    # Direct Plan
    ax_direct_v = plt.subplot(332)
    ax_direct_v.plot(t_direct, s_v_mag_direct, color=colors_fig[2])
    ax_direct_v.set_xlim([0 - t_direct_10pct, t_direct_end + t_direct_10pct + a_end_buffer])
    ax_direct_v.set_ylim(v_range)
    ax_direct_v.set_xticks([], [])

    ax_direct_a = plt.subplot(335)
    ax_direct_a.plot(t_direct, s_a_mag_direct, color=colors_fig[2])
    ax_direct_a.plot([0, 0], [0, 100], color=colors_fig[2])
    ax_direct_a.plot([1.45, 1.45], [0, 100], color=colors_fig[2])
    ax_direct_a.plot([2.9, 2.9], [0, 100], color=colors_fig[2])
    ax_direct_a.set_xlim([0 - t_direct_10pct, t_direct_end + t_direct_10pct + a_end_buffer])
    ax_direct_a.set_ylim(a_range)
    ax_direct_a.set_xticks([], [])

    ax_direct_lambda = plt.subplot(338)
    ax_direct_lambda.plot(t_cv_direct, cv_direct, color=colors_fig[2])
    ax_direct_lambda.set_xlim([0 - t_direct_10pct, t_direct_end + t_direct_10pct + a_end_buffer])
    ax_direct_lambda.set_ylim([-0.75, 65])

    # Feasible Plan
    ax_feasible_v = plt.subplot(333)
    ax_feasible_v.plot(t_feasible, s_v_mag_feasible, color=colors_fig[3])
    ax_feasible_v.set_xlim([0 - t_feasible_10pct, t_feasible_end + t_feasible_10pct + a_end_buffer])
    ax_feasible_v.set_ylim(v_range)
    ax_feasible_v.set_xticks([], [])

    ax_feasible_a = plt.subplot(336)
    ax_feasible_a.plot(t_feasible, s_a_mag_feasible, color=colors_fig[3])
    ax_feasible_a.set_xlim([0 - t_feasible_10pct, t_feasible_end + t_feasible_10pct + a_end_buffer])
    ax_feasible_a.set_ylim(a_range)
    ax_feasible_a.set_xticks([], [])

    ax_feasible_lambda = plt.subplot(339)
    ax_feasible_lambda.plot(t_cv_feasible, cv_feasible, color=colors_fig[3])
    ax_feasible_lambda.set_xlim([0 - t_feasible_10pct, t_feasible_end + t_feasible_10pct + a_end_buffer])
    ax_feasible_lambda.set_ylim([-0.75, 65])


    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig('fig_ga_val_transparent.png', transparent=True)
    print(' ')
