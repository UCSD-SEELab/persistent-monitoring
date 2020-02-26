#!/usr/bin/env python3
"""
main_sim.py
Michael Ostertag

A simulation environment to test persistent monitoring algorithms using robotic platforms.

The simulation keeps track of an environmental model (env_model) and drone states (swarm_controller).
"""

import time
import argparse

import os
import sys
import math
import logging
import numpy as np
import csv
from env_models import Model_Base
from drones import Drone_Base, Drone_Ostertag2020, Drone_Constant, Drone_Ostertag2019, Drone_Smith2012
from shapely.geometry import Polygon  # , LineString,  Point
# from shapely.geometry.polygon import orient

import matplotlib.cm as colors
import matplotlib.pyplot as plt

####################
class Sim_Environment():
    """
    Simulation environment for single robotic platforms measuring N random points of interest
    """

    def __init__(self, swarm_controller=None, env_model=None, step_time=1,
                 b_verbose=True, b_logging=True):
        """
        """

        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.step_time = step_time

        self.swarm_controller = swarm_controller
        if (swarm_controller == None):
            self.b_swarm_controller = False
        else:
            self.b_swarm_controller = True

        self.env_model = env_model
        if (self.env_model == None):
            self.b_env_model = False
        else:
            self.b_env_model = True

        self.filename_results = time.strftime('Result_%Y%m%d_%H%M%S', time.localtime())

        plt.ion()  # Enable interactive plotting

    def set_swarm_controller(self, swarm_controller):
        """
        Links a swarm controller to the simulation environment after initialization
        """
        self.swarm_controller = swarm_controller
        self.b_swarm_controller = True

        if (self.b_env_model):
            self.swarm_controller.set_env_model( self.env_model )

    def set_env_model(self, env_model):
        """
        Links an environmental model to the simulation
        """
        self.env_model = env_model
        self.b_env_model = True

        if (self.b_swarm_controller):
            self.swarm_controller.set_env_model( self.env_model )
            self.swarm_controller.calc_trajectory_init()

    def init_sim(self):
        """
        Initializes the simulation
        """
        if (self.env_model):
            self.env_model.init_sim()

        if (self.swarm_controller):
            self.swarm_controller.init_sim()

    def update(self):
        """
        description
        """
        if (self.env_model):
            self.env_model.update( dt=self.step_time )

        if (self.swarm_controller):
            self.swarm_controller.update( dt=self.step_time)

    def save_results(self, param_ros=0,
                     param_covar_0_scale=0,
                     param_v_max=0,
                     param_n_obs=0,
                     param_N_q=0,
                     theta0=0):
        """
        Save the configuration results from the environmental model and the
        covariance results from the individual drones to a csv file
        """
        list_config = self.env_model.get_config()
        list_results = self.swarm_controller.get_results()

        np.set_printoptions(linewidth=1024, suppress=True)

        filename_temp = '{0}_vmax{2:d}_V{3:d}_N{4:d}_ros{1:d}.csv'.format(
            self.filename_results, math.trunc(param_ros),
            math.trunc(param_v_max), math.trunc(param_n_obs),
            math.trunc(param_N_q))
        directory_save = '../results/'
        if not os.path.exists(directory_save):
            os.makedirs(directory_save)
        with open(directory_save + filename_temp, 'a', newline='') as fid:
            writer = csv.writer(fid)
            for result in list_results:
                temp_row = list_config + [theta0] + list(result.flatten())
                writer.writerow(temp_row)

    def visualize(self):
        plt.clf()
        plt.figure(1)
        plt.subplot(111)

        if (self.b_env_model):
            self.env_model.visualize()

        if (self.b_swarm_controller):
            self.swarm_controller.visualize()

        plt.draw()
        plt.pause(0.01)


####################
class SwarmController():
    """
    SwarmController controls the communication between all drones and manages
    their movement calculation, updates, and visualizations
    """

    def __init__(self, env_model=None,
                 b_verbose=True, b_logging=True):
        """
        gamma, poi, covar_e, map_terrain=np.matrix([]), map_data=np.matrix([]),
                 N_drones=1, pos_start=0, list_pos_start=[], step_size=1, step_time=1,
                 param_covar_0_scale=100, param_v_max=10, param_n_obs=5, param_ros=1,
                 b_verbose=True, b_logging=True):
        """
        """
        Initializes the swarm controller
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging

        self.env_model = env_model
        if (self.env_model == None):
            self.b_env_model = False
        else:
            self.b_env_model = True

        self.list_drones = []

    def add_drone(self, drone_type, drone_id, b_verbose=True, b_logging=True, cfg=None):
        """
        Adds a drone specified by drone_type, which should be a drone model that conforms to the standard in drones.py.
        cfg is a dictionary that holds the configuration of the drone for all required parameters by the class.
        """
        # try:
        temp_drone = drone_type(drone_id=drone_id, b_verbose=b_verbose, b_logging=b_logging, **cfg)
        self.list_drones.append(temp_drone)
        print('Drone ({0}) added to swarm_controller'.format(drone_type.__name__))
        """except:
            print(sys.exc_info())
            print('Drone ({0}) failed to be created with parameters {1}'.format(drone_type.__name__, cfg))
        """

    def init_sim(self):
        """
        Initialize each drone for simulation that is about to start
        """

        for drone in self.list_drones:
            drone.init_sim()

    def update(self, dt):
        """
        description
        """
        temp_covar_max = np.zeros((len(self.list_drones), 1))

        for ind, drone in enumerate(self.list_drones):
            drone.update( dt )
            temp_covar_max[ind, 0] = np.max(np.diag(drone.get_covar_s()))

        # self.list_covar_max = np.append(self.list_covar_max, temp_covar_max, axis=1)

    def reset_covar_max(self):
        """
        Reset self.list_covar_max to initial value
        """
        self.list_covar_max = self.list_covar_max_0[:]

    def set_env_model(self, env_model):
        """
        Sets the environmental model and resets all drones
        """
        self.env_model = env_model
        # for drone in self.list_drones:
        #     drone.set_terrain_map(self.map_terrain) # Drones not currently using map

    def get_results(self):
        """
        Return results of the drone IDs and their covariance values
        """
        return self.list_covar_max

    def reset_drone(self, theta0, covar_0_scale=100):
        """
        Resets the drone to a provided theta0 with some initial
        covariance scaled by covar_0_scale while maintaining any generated
        models
        """
        self.reset_covar_max()
        for ind, drone in enumerate(self.list_drones):
            drone.init_covar(covar_0_scale)
            drone.set_theta(theta0)

    def visualize(self):
        for ind, drone in enumerate(self.list_drones):
            drone.visualize(colors.get_cmap('tab10')(ind % 10))


####################
"""
Main control loop
"""
if __name__ == '__main__':
    """
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--vmax', type=float, default=30.0,
                        help='maximum velocity of drone')
    parser.add_argument('--v', type=float, default=20.0,
                        help='noise in observation model')
    parser.add_argument('--n', type=int, default=6,
                        help='number of points of interest (q)')
    parser.add_argument('--Nsteps', type=int, default=600,
                        help='number of simulation steps')
    parser.add_argument('--Ntests', type=int, default=100,
                        help='number of independent tests')
    parser.add_argument('--Nrobust', type=int, default=1,
                        help='number of repetitions at different initial positions')
    parser.add_argument('--verbose', action='store_true',
                        help='enables verbose output')
    parser.add_argument('--logging', action='store_true',
                        help='enables debug logging')
    parser.add_argument('--overlap', action='store_true',
                        help='forces at least two points of interest to have overlapping sensing region')
    args = parser.parse_args()

    N_steps = args.Nsteps
    N_drones = 1
    env_size = [500, 500]
    step_size = 1  # 1 meter
    step_time = 1  # 1 second

    N_tests = args.Ntests
    N_robust = args.Nrobust
    param_ros = 1
    param_covar_0_scale = 100
    param_v_max = args.vmax
    param_n_obs = args.nobs
    param_N_q = args.Nq

    b_verbose = args.verbose
    b_logging = args.logging
    b_overlap = args.overlap
    """

    b_logging = False
    b_verbose = True
    N_tests = 1
    step_size = 1
    step_time = 0.1

    # env_model
    env_size = np.array([100, 100])
    param_N_q = 10

    if (b_logging):
        logger_filename = time.strftime('Log_%Y%m%d_%H%M%S.log', time.localtime())
        # TODO If Logs doesn't exist, create folder.
        logger_format = "[%(asctime)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
        logging.basicConfig(filename='Logs/' + logger_filename, level=logging.DEBUG,
                            format=logger_format, datefmt='%Y%m%d %H%M%S')
        logger = logging.getLogger('root')

    # Configure simulation environment and initialize
    sim_env = Sim_Environment(swarm_controller=None, env_model=None, step_time=step_time,
                 b_verbose=True, b_logging=True)

    for ind_test in range(N_tests):
        print('Test {0:4d}/{1}'.format(ind_test + 1, N_tests))

        # Setup environmental model
        env_model = Model_Base(env_size=env_size, step_size=step_size, N_q=param_N_q,
                                              b_verbose=b_verbose, b_logging=b_logging)
        sim_env.set_env_model(env_model)

        # Set up swarm controller and drones to test
        swarm_controller = SwarmController(env_model=env_model, b_verbose=b_verbose, b_logging=b_logging)
        sim_env.set_swarm_controller(swarm_controller)
        swarm_controller.add_drone(drone_type=Drone_Constant, drone_id='Drone1', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':25, 'fs':2})
        swarm_controller.add_drone(drone_type=Drone_Ostertag2020, drone_id='Drone2', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax':25, 'fs':2})
        swarm_controller.add_drone(drone_type=Drone_Ostertag2019, drone_id='Drone3', b_verbose=b_verbose, b_logging=b_logging,
                                    cfg={'env_model': env_model, 'vmax':25, 'fs':2})
        swarm_controller.add_drone(drone_type=Drone_Smith2012, drone_id='Drone4', b_verbose=b_verbose, b_logging=b_logging,
                                   cfg={'env_model': env_model, 'vmax': 25, 'fs': 2})

        # Everything is connected, initialize
        sim_env.init_sim()

        for ind in range(1000):
            sim_env.update()

            # Update visualization
            # if ((ind_loop >= 2) and ((ind_loop % 5) == 0)):
            #     sim_env.visualize()
            sim_env.visualize()
            time.sleep(0.01)

    if (b_logging):
        logging.shutdown()
