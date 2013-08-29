import momo
import numpy as np
from math import *
from __common__ import *

def max_idx( value, reference ):
  cur_idx = 0
  for i in xrange( len( reference ) ):
    if value >= reference[i]:
      cur_idx = i
  return cur_idx
      

def compute_feature( reference, frame, radius = 3 ):
  density = 0
  feature = np.array( [0.] * FEATURE_LENGTH, dtype = np.float32 )
  avg_velocity = np.array( [0.] * 2, dtype = np.float32 )

  for i in xrange( len( frame ) ):
    dist  = momo.distance( frame[i][:2], reference[:2] )
    if dist < radius:
      density += 1
      avg_velocity += frame[i][2:]

  if density > 0:
    velocity = reference[2:]

    avg_velocity /= density
    avg_velocity  = velocity - avg_velocity

    feature[dns_i + max_idx( density, DENSITIES )] = 1

    speed = np.linalg.norm( avg_velocity )
    feature[spd_i + max_idx( speed, SPEEDS )] = 1

    cosine = np.dot( avg_velocity / speed, velocity / np.linalg.norm( velocity ) )
    feature[dir_i + max_idx( cosine, ANGLES )] = 1

    
  return feature



