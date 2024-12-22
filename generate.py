# %% Modules

import os
import glob
import numpy as np
from scipy.spatial import distance_matrix

# %% Generate random set of cities

num_points = 16

x_min = 0.0
y_min = 0.0
x_max = 1.0
y_max = 1.0

x = np.random.uniform(x_min, x_max, num_points)
y = np.random.uniform(y_min, y_max, num_points)

points = np.column_stack((x, y))
distances = distance_matrix(points, points)

# %% Save data

data_id = int(sorted(glob.glob("data/*"))[-1].split("data/p")[-1]) + 1
path = "data/p%d" % (data_id)
os.mkdir(path)
np.savetxt("%s/locations.dat" % (path), points)
np.savetxt("%s/distances.dat" % (path), distances)

# %% End of program
