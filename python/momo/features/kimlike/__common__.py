import numpy as np
from math import *

FEATURE_LENGTH = 9

DIRECTIONS = np.array( [
  [ 1.,  0.],
  [ 1.,  1.],
  [ 0.,  1.],
  [-1.,  1.],
  [-1.,  0.],
  [-1., -1.],
  [ 0., -1.],
  [ 1., -1.]
], dtype = np.float32 )

DENSITIES  = np.array( [ 0., 2.0, 5.0], dtype = np.float32 )
SPEEDS     = np.array( [0.0, 0.015, 0.025], dtype = np.float32 )
ANGLES     = np.array( [cos( pi ), cos( 5 * pi / 6 ), cos( pi / 6 )], dtype = np.float32 )
DISTANCE   = np.array( [0., 1., 2.], dtype = np.float32 )

dns_i = 0
spd_i = 3
dir_i = 6
dst_i = 9
