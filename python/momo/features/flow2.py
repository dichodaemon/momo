from momo import *
from default import *

import numpy as np
from math import *

densities  = [3, 2, 1, 0]
density_2  = [3, 1, 0]
angles     = [7 * pi / 8, 5 * pi / 8, 3 * pi / 8, pi / 8]


n_density   = len( densities )
n_density_2 = len( density_2 )
n_angle     = len( angles )
feature_size = n_density + n_density_2 * n_angle + 1
default = np.array( [0.] * feature_size )

def compute( s1, s2, frame ):
  result = default * 1.
  agt_vel = s2[2:]
  tot_density = 0
  for o in frame:
    dist  = distance( s2[:2], o[:2] ) * 3.5
    if dist < 4:
      tot_density += 1
      v1 = agt_vel
      v2 = o[2:]
      n1 = np.dot( v1, v1 )
      n2 = np.dot( v2, v2 )
      if n1 > 0:
        v1 /= n1
      else:
        v1 *= 0.
      if n2 > 0:
        v2 /= n2
      else:
        v2 *= 0.
      r_vel = v2 - v1 
      r_angle = angle.as_angle( r_vel )
      i_angle = 0
      min_angle = 1E6
      for i in xrange( n_angle ):
        ang = abs( angle.difference( r_angle, angles[i] ) )
        if ang < min_angle:
          i_angle = i
          min_angle = ang
      result[n_density + i_angle * n_density_2] += 1
  for i in xrange( n_density ):
    if tot_density >= densities[i]:
      result[i] = 1
      break
  for i_angle in xrange( n_angle ):
    for i_density in xrange( n_density_2 ):
      if result[n_density + i_angle * n_density_2] >= density_2[i_density]:
        result[n_density + i_angle * n_density_2] = 0
        result[n_density + i_angle * n_density_2 + i_density] = 1
        break
  result[n_density + n_density_2 * n_angle] = distance( s1[:2], s2[:2] )
  return result

