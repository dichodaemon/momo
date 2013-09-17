import numpy as np
from math import *

FEATURE_LENGTH = 18

ANGLES = np.array( [
  [cos( pi / 8 ), sin( pi / 8 )],
  [cos( 3 * pi / 8 ), sin( 3 * pi / 8 )],
  [cos( 5 * pi / 8 ), sin( 5 * pi / 8 )],
  [cos( 7 * pi / 8 ), sin( 7 * pi / 8 )],
], dtype = np.float32 )

SPEEDS = np.array( [0., 0.015, 0.03], dtype = np.float32 )
