from momo import *
from default import *

import numpy as np
from math import *

densities  = [3, 2, 1, 0]
velocities = [0.05, 0.02, 0.]
angles     = [7 * pi / 8, 5 * pi / 8, 3 * pi / 8, pi / 8]

n_density  = len( densities )
n_velocity = len( velocities )
n_angle    = len( angles )
feature_size = n_density + n_velocity * n_angle + 1
default = np.array( [0.] * feature_size )

def compute( s1, s2, frame ):
  result = default * 1.
  agt_vel = s2[2:]
  tot_density = 0
  sum_vel = np.array( [0.] * 2 )
  for o in frame:
    dist  = distance( s2[:2], o[:2] )
    r_vel   = o[2:] - agt_vel
    if dist < 1:
      tot_density += 1
      sum_vel += r_vel
  if tot_density > 0:
    sum_vel /= tot_density
    r_angle = abs( angle.as_angle( sum_vel ) )
    r_speed = np.dot( sum_vel, sum_vel ) ** 0.5
    for i in xrange( n_density ):
      if tot_density >= densities[i]:
        result[i] = 1
        break
    i_speed = 0
    for i in xrange( n_velocity ):
      if r_speed >= velocities[i]:
        i_speed = i
        break
    i_angle = 0
    min_angle = 1E6
    for i in xrange( n_angle ):
      ang = abs( angle.difference( r_angle, angles[i] ) )
      if ang < min_angle:
        i_angle = i
        min_angle = ang
    result[n_density + i_angle * n_velocity + i_speed] = 1
  else:
    result[n_density - 1] = 1
    result[n_density + n_velocity * n_angle] = 1
  #result[n_density + n_velocity * n_angle] = distance( s1[:2], s2[:2] )
  return result

