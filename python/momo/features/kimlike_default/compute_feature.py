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
  density = 0.0
  feature = np.array( [0.] * FEATURE_LENGTH, dtype = np.float32 )
  sum_ang = 0.0
  sum_mag = 0.0

  for i in xrange( len( frame ) ):
    dist  = momo.distance( frame[i][:2], reference[:2] )
    if dist < radius:
      density += 1
      rel_v = frame[i][2:] - reference[2:]
      rel_x = frame[i][:2] - reference[:2]
      l_v = np.linalg.norm( rel_v )
      l_x = np.linalg.norm( rel_x )
      a = np.dot( rel_v / l_v, rel_x / l_x )
      sum_ang += a
      sum_mag += l_v

  if density > 0:
    feature[dns_i + max_idx( density, DENSITIES )] = 1

    speed = sum_mag / density
    feature[spd_i + max_idx( speed, SPEEDS )] = 1

    angle = sum_ang / density
    feature[dir_i + max_idx( angle, ANGLES )] = 1
  feature[FEATURE_LENGTH - 1] = 1
    
  return feature



