"""
env_models.py

Contains environmental models for testing with drones from drones.py in 
MinimizingUncertainty project:
    
    Model_StaticFire: static fire model on flat terrain

"""

import time

import math
import logging
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LinearRing, LineString, Point, MultiPoint
from shapely.ops import unary_union
from matplotlib.path import Path
import csv

import matplotlib.cm as colors
import matplotlib.pyplot as plt

from noise import pnoise2


####################
"""
Helper functions
"""
def dist_l1(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sum(np.abs(x[0:ind_end] - y[0:ind_end]))


def dist_l2(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sqrt(np.sum(np.power(x[0:ind_end] - y[0:ind_end], 2)))


####################
class Model_Base():
    """
    Base class for environmental models that holds all basic functionality
    """
    def __init__(self, env_size=np.array([100,100]), N_q=10, step_size=1,
                 b_terrain=False, b_verbose=True, b_logging=True):
        """
        Initializing the base for the models requires at a minimum:

        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.b_terrain = b_terrain

        self.env_size = env_size
        self.step_size = step_size

        self.N_q = N_q

        self.logger = logging.getLogger('root')
        
        # Randomly initialize points of interest with non-overlapping sensing regions
        self.init_pois()

        # Generate terrain, using Perlin noise if more beautiful appearance is desired
        self.generate_terrain(b_perlin=self.b_terrain)

    def init_pois(self):
        """
        Initialize the points of interest x and associated covariance of growth
        """
        # TO BE UPDATED FOR SPECIFIC MODELS
        theta = np.linspace(0, 2*math.pi, self.N_q + 1)
        r = 35
        x = -r * np.cos(theta[:-1]).reshape((-1,1)) + self.env_size[0] / 2
        y = r * np.sin(theta[:-1]).reshape((-1,1)) + self.env_size[1] / 2

        self.list_q = np.append(x, y, axis=1)

        self.covar_env = np.diag(5 + 2.5*np.cos( theta[:-1] )) / 10

    def get_map_terrain(self):
        """
        Gets the object reference to the terrain map
        """
        return self.map_terrain

    def generate_terrain(self, b_perlin=True):
        """
        Generates terrain using Perlin noise from Python package noise 1.2.2
        (https://pypi.python.org/pypi/noise/)

         pnoise2 = noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0,
                          repeatx=1024, repeaty=1024, base=0.0)

         return perlin "improved" noise value for specified coordinate

         octaves -- specifies the number of passes for generating fBm noise,
             defaults to 1 (simple noise).

         persistence -- specifies the amplitude of each successive octave relative
             to the one below it. Defaults to 0.5 (each higher octave's amplitude
             is halved). Note the amplitude of the first pass is always 1.0.

         lacunarity -- specifies the frequency of each successive octave relative
             to the one below it, similar to persistence. Defaults to 2.0.

         repeatx, repeaty, repeatz -- specifies the interval along each axis when
             the noise values repeat. This can be used as the tile size for creating
             tileable textures

         base -- specifies a fixed offset for the input coordinates. Useful for
             generating different noise textures with the same repeat interval
        """

        self.map_terrain = np.zeros((self.env_size))

        scale_x = 0.1
        offset_x = 10
        scale_y = 0.1
        offset_y = 10
        scale_alt = 10
        offset_alt = 1000
        if (b_perlin):
            for val_x in range(self.env_size[0]):
                for val_y in range(self.env_size[1]):
                    x = scale_x * val_x + offset_x
                    y = scale_y * val_y + offset_y
                    self.map_terrain[val_x, val_y] = pnoise2(x, y, octaves=4, persistence=1.5,
                                                             lacunarity=0.5) * scale_alt + offset_alt
        else:
            self.map_terrain += offset_alt

        self.x_plt, self.y_plt = np.meshgrid(range(self.env_size[0]), range(self.env_size[1]))

        if (self.b_verbose):
            print('Min: {0:0.2f} m   Max: {1:0.2f} m'.format(self.map_terrain.min(), self.map_terrain.max()))

    def set_terrain_map(self, map_in):
        """
        Provides an object reference to a terrain map for use in environmental
        models
        """
        self.map_terrain = map_in
    
    def get_pois(self):
        """
        Returns underlying state positions
        """
        return self.list_q
    
    def get_covar_env(self):
        """
        Returns covariance matrix of Wiener process
        """
        return self.covar_env

    def init_sim(self):
        """
        Initialize any required elements for the simulation
        """
        pass

    def update(self, dt):
        """
        Update model for changes over dt time
        """
        # REPLACE WITH REQUIRED CODE IF NONSTATIC MODEL
        pass
    
    def visualize(self):
        """
        Visualizes the points of interest (list_q) and underlying terrain
        """
        # Make something pretty like http://www.harrisgeospatial.com/docs/EM1_SurfacePlot.html
        # https://matplotlib.org/basemap/users/examples.html

        ax = plt.gca()

        # cmap_fire = colors.get_cmap('hot')
        # cmap_fire.set_under('k', alpha=0)
        # 
        # plt.pcolor(self.list_x[:, 0], self.list_x[:, 1], 
        #            np.diag(self.covar_env), vmin=0.01, vmax=1.5, 
        #            cmap=cmap_fire, zorder=1)
        # plt.colorbar()

        if (self.b_terrain):
            plt.pcolor(self.x_plt, self.y_plt, self.map_terrain,
                       cmap='Greens_r')

        ax.scatter(self.list_q[:, 0], self.list_q[:, 1], color='black', marker='o')
        ax.set_xlim([0, self.env_size[0]])
        ax.set_ylim([0, self.env_size[1]])

####################
class Model_Randomized(Model_Base):
    """
    Randomized non-overlapping points of interest
    """
    def __init__(self, env_size=np.array([100,100]), N_q=10, step_size=1, B=10,
                 b_terrain=False, b_verbose=True, b_logging=True):
        """
        Initializes an environmental model that has N_q randomly initialized points
        """
        # Sensing radius of the robotic system
        self.B = B

        super(Model_Randomized, self).__init__(env_size=env_size, N_q=N_q, step_size=step_size,
                                               b_terrain=b_terrain, b_verbose=b_verbose, b_logging=b_logging)

    def init_pois(self):
        """
        Initialize the points of interest x spread across a grid in random fashion and uncertainty growth rates with a
        chi-squared distribution
        """
        b_searching = True
        arr_q = np.zeros((self.N_q, 2))

        while (b_searching):
            # Initialize grid of all possible points
            x_pos = np.arange(self.B + 1, self.env_size[0] - self.B - 1)
            y_pos = np.arange(self.B + 1, self.env_size[0] - self.B - 1)

            x_mesh, y_mesh = np.meshgrid(x_pos, y_pos)

            xy_ind = np.arange(len(x_pos) * len(y_pos))
            xy_valid = np.ones(xy_ind.shape).astype(bool)

            for ind_q in range(self.N_q):
                # Choose a single valid location
                ind_point = np.random.choice(xy_ind[xy_valid])
                x_ind = ind_point % len(x_pos)
                y_ind = ind_point // len(x_pos)

                # Add to the array
                x = x_pos[x_ind]
                y = y_pos[y_ind]
                arr_q[ind_q, 0], arr_q[ind_q, 1] = x, y

                # Invalidate all other points within range
                mesh_valid = np.power(x_mesh - x, 2) + np.power(y_mesh - y, 2) > (2 * self.B)**2
                xy_valid = np.logical_and(xy_valid, mesh_valid.flatten())

                if ind_q == (self.N_q - 1):
                    b_searching = False

                if all(xy_valid == False):
                    break

        print('Locations remaining: {0:0.2f}%'.format(100 *np.sum(xy_valid) / np.product(self.env_size)))

        self.list_q = arr_q

        self.covar_env = np.diag((2*np.random.random(size=self.N_q))**2)

####################
"""
Test functionality by generating and plotting an environment
"""
if __name__ == '__main__':
    list_N_q = [10, 20, 40, 60, 80, 100]
    n_iter = 3

    for N_q in list_N_q:
        for _ in range(n_iter):
            test_env = Model_Randomized(env_size=np.array([450, 450]), N_q=N_q, step_size=1,
                         b_terrain=False, b_verbose=False, b_logging=False, B=10)

            #tempx, tempy = np.meshgrid(range(test_env.env_size[0]), range(test_env.env_size[1]))
            plt.figure(1)
            plt.subplot(111)
            # plt.pcolor(tempx, tempy, test_env.map_terrain,
            #            cmap='Greens_r')  # Can use 'terrain' colormap once elevation is normalized
            # plt.colorbar()
            plt.title('{0:2d} POI'.format(N_q))

            test_env.visualize()
            plt.show()

    print(' ')