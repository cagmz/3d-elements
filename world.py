"""
The World contains a Neighborhood of cells for each of the elements (fire, water, cloud, vegetation).
Cells within four 'neighborhoods' interact with each other, while also influencing other neighborhoods.

The Ecosystem is toroidal (2D array wraps around) and uses the Moore neighborhood.
So cell (0, 0) has neighbors [(4, 4), (0, 4), (1, 4), (4, 0), (1, 0), (4, 1), (0, 1), (1, 1)].

Double-buffering is used to for interactions between neighborhoods.
"""

# 3d plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
# colormaps
import matplotlib.cm as cm
# general plotting
import matplotlib.pyplot as plt
import numpy as np
from neighborhood import Neighborhood


def normalize_cells(level):
    return (level.cells - level.relative_min) / (level.relative_max - level.relative_min)


def sigmoidal(x, minimum=0, maximum=10, slope=1):
    return maximum / (1 + np.exp(-slope * (x - (maximum - minimum) / 2)))


def gt_threshold(array, threshold):
    """
    Returns a list of tuples containing which coordinates are > the given threshold.
    Used when checking if a cloud cell can make rain, or if a fire cell is active.
    """
    x, y = np.nonzero(np.array(array > threshold))
    coordinates = list(zip(x, y))
    return coordinates


def lt_threshold(array, threshold):
    """
    Returns a list of tuples containing which coordinates are < the given threshold.
    Used when checking if a veg cell is dry.
    """
    x, y = np.nonzero(np.array(array < threshold))
    coordinates = list(zip(x, y))
    return coordinates


class World:
    def __init__(self, config, debug=False):
        self.debug = debug
        self.config = config

        # neighborhoods
        self.cloud = None
        self.fire = None
        self.veg = None
        self.water = None

        self.neighborhoods = config['neighborhoods']

        self.m, self.n = int(config['dimensions']['m']), int(config['dimensions']['n'])
        self.size = (self.m, self.n)

        # graphing
        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = None, None
        self.X, self.Y = None, None
        self.init_neighborhoods()
        self.init_figure()
        self.plot()

    def init_figure(self):
        self.fig = self.plt.figure(figsize=self.size)
        self.ax = self.fig.gca(projection='3d')
        X = np.arange(0, self.m, 1)
        Y = np.arange(0, self.n, 1)
        self.X, self.Y = np.meshgrid(X, Y)

    def init_neighborhoods(self):
        self.water = Neighborhood(self.neighborhoods['water']['min_z'], self.neighborhoods['water']['max_z'],
                                  size=self.size)
        self.cloud = Neighborhood(self.neighborhoods['cloud']['min_z'], self.neighborhoods['cloud']['max_z'],
                                  size=self.size)
        self.veg = Neighborhood(self.neighborhoods['veg']['min_z'], self.neighborhoods['veg']['max_z'], size=self.size)
        self.fire = Neighborhood(self.neighborhoods['fire']['min_z'], self.neighborhoods['fire']['max_z'],
                                 size=self.size)

        if not self.debug:
            return

        print(
            'Initialized neighborhoods...\nWater cells:\n{}\nCloud cells:\n{}\nVeg cells:\n{}\nFire cells:\n{}'.format(
                self.water.cells_copy, self.cloud.cells_copy, self.veg.cells_copy, self.fire.cells_copy))

    def get_neighbors(self, pair):
        i, j = pair
        # When a cell is at a fringe do wrap
        iprev = i - 1 if i > 0 else self.m - 1
        inext = i + 1 if i < self.m - 1 else 0
        jprev = j - 1 if j > 0 else self.n - 1
        jnext = j + 1 if j < self.n - 1 else 0

        return [(iprev, jprev), (i, jprev), (inext, jprev),  # cells above
                (iprev, j), (inext, j),  # cells either side
                (iprev, jnext), (i, jnext), (inext, jnext)]  # cells below

    def water_tick(self):
        self.water.cells_copy = self.water.cells.copy()

        # water loss from evaporation
        self.water.cells_copy -= self.water.cells_copy * self.neighborhoods['water']['evaporation_rate']

        # add rain to main water cell and to neighbors
        raining_cells = gt_threshold(normalize_cells(self.cloud), self.neighborhoods['cloud']['rain_threshold'])
        for cloud in raining_cells:
            main_rainfall = self.water.cells_copy[cloud] * self.neighborhoods['cloud']['rainfall']
            self.water.cells_copy[cloud] += main_rainfall
            neighbor_rainfall = sigmoidal(main_rainfall)
            water_cell_neighbors = self.get_neighbors(cloud)
            for neighbor in water_cell_neighbors:
                self.water.cells_copy[neighbor] += neighbor_rainfall

        # subtract veg: all veg cells draw water
        self.water.cells_copy -= self.water.cells_copy * self.neighborhoods['veg']['water_draw']

        np.clip(self.water.cells_copy, self.water.relative_min, self.water.relative_max, self.water.cells_copy)

    def cloud_tick(self):
        self.cloud.cells_copy = self.cloud.cells.copy()
        # increase cloud level according to water level
        self.cloud.cells_copy += self.cloud.cells_copy * self.neighborhoods['water']['evaporation_rate']
        raining_cells = gt_threshold(normalize_cells(self.cloud), self.neighborhoods['cloud']['rain_threshold'])
        for cloud in raining_cells:
            main_water_loss = self.cloud.cells_copy[cloud] * self.neighborhoods['cloud']['rainfall']
            self.cloud.cells_copy[cloud] -= main_water_loss
            neighbor_water_loss = sigmoidal(main_water_loss)
            cloud_neighbors = self.get_neighbors(cloud)
            for neighbor in cloud_neighbors:
                self.cloud.cells_copy[neighbor] -= neighbor_water_loss

        np.clip(self.cloud.cells_copy, self.cloud.relative_min, self.cloud.relative_max, self.cloud.cells_copy)

    def veg_tick(self):

        self.veg.cells_copy = self.veg.cells.copy()

        # add to veg from water cells, if available
        available_water_cells = gt_threshold(normalize_cells(self.water),
                                             self.neighborhoods['water']['available_threshold'])
        for water in available_water_cells:
            main_water_draw = self.veg.cells_copy[water] * self.neighborhoods['veg']['water_draw']
            self.veg.cells_copy[water] += main_water_draw
            neighbor_draw = sigmoidal(main_water_draw)
            veg_neighbors = self.get_neighbors(water)
            for neighbor in veg_neighbors:
                self.veg.cells_copy[neighbor] += neighbor_draw

        no_water_cells = lt_threshold(normalize_cells(self.water), self.neighborhoods['water']['available_threshold'])
        for no_water in no_water_cells:
            self.veg.cells_copy[no_water] -= self.veg.cells_copy[no_water] * self.neighborhoods['veg']['water_draw']

        active_fires = gt_threshold(normalize_cells(self.fire), self.neighborhoods['fire']['fire_threshold'])
        for fire in active_fires:
            self.veg.cells_copy[fire] = self.veg.relative_min

        np.clip(self.veg.cells_copy, self.veg.relative_min, self.veg.relative_max, self.veg.cells_copy)

    def fire_tick(self):
        self.fire.cells_copy = self.fire.cells.copy()

        # normal plus rain decay
        self.fire.cells_copy -= self.fire.cells_copy * self.neighborhoods['fire']['decay']
        raining_cells = gt_threshold(normalize_cells(self.cloud), self.neighborhoods['cloud']['rain_threshold'])
        for rain_cell in raining_cells:
            self.fire.cells_copy[rain_cell] -= self.fire.cells_copy[rain_cell] * self.neighborhoods['fire'][
                'rain_decay']

        # find a spot for lightning
        lightning_strike = np.random.choice(self.m), np.random.choice(self.n)
        dry_vegetation = lt_threshold(normalize_cells(self.veg), self.neighborhoods['veg']['dry_threshold'])
        # only start new fire if 0 < veg_level < dry_threshold
        if lightning_strike in dry_vegetation and normalize_cells(self.veg)[lightning_strike]:
            self.fire.cells_copy[lightning_strike] = self.fire.relative_max
            neighbors = self.get_neighbors(lightning_strike)
            for neighbor in neighbors:
                if 0 < self.veg.cells[neighbor] < self.neighborhoods['veg']['dry_threshold']:
                    self.fire.cells_copy[neighbor] = self.fire.relative_max

        np.clip(self.fire.cells_copy, self.fire.relative_min, self.fire.relative_max, self.fire.cells_copy)

    def tick(self):
        self.water_tick()
        self.cloud_tick()
        self.veg_tick()
        self.fire_tick()

        # after all level ticks, update cells and then plot
        self.water.cells = self.water.cells_copy
        self.cloud.cells = self.cloud.cells_copy
        self.veg.cells = self.veg.cells_copy
        self.fire.cells = self.fire.cells_copy

        if not self.debug:
            return

        print('\n\t\tNew Tick:\nCloud:\n{}\nFire:\n{}\nVeg:\n{}\nWater:\n{}\n'.format(
            normalize_cells(self.cloud),
            normalize_cells(self.fire),
            normalize_cells(self.veg),
            normalize_cells(self.water)))

    def plot(self):
        self.fig.clear()
        self.ax = self.fig.gca(projection='3d')
        self.plt.tick_params(labelbottom='off', labelleft='off')
        self.ax.plot_surface(self.X, self.Y, self.cloud.cells,
                             rstride=1, cstride=1, linewidth=0,
                             cmap=plt.get_cmap('Greys', 256),
                             alpha=np.mean(normalize_cells(self.cloud)))
        self.ax.plot_surface(self.X, self.Y, self.fire.cells,
                             rstride=1, cstride=1, linewidth=0,
                             cmap=plt.get_cmap('Reds', 256),
                             alpha=np.mean(normalize_cells(self.fire)))
        self.ax.plot_surface(self.X, self.Y, self.veg.cells,
                             rstride=1, cstride=1, linewidth=0,
                             cmap=plt.get_cmap('Greens', 256),
                             alpha=np.mean(normalize_cells(self.veg)))
        self.ax.plot_surface(self.X, self.Y, self.water.cells,
                             rstride=1, cstride=1, linewidth=0,
                             cmap=plt.get_cmap('Blues', 256),
                             alpha=np.mean(normalize_cells(self.water)))
        self.fig.canvas.draw()
