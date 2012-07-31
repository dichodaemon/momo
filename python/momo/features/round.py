from momo import *

import numpy as np
from math import *

feature_size = 5

def compute( reference, frame ):
  angles    = [0, pi / 4, 3 * pi / 4, -pi / 4, -3 * pi / 4]
  distances = [1., 2., 4., 100.]
  result = np.array( [0.] * len( angles ) )
  r_angle = angle.as_angle( reference[2:] )
  for o in frame:
    o_angle = angle.as_angle( o[:2] - reference[:2] )
    t_angle = angle.difference( r_angle, o_angle )
    t_distance = distance( o[:2], reference[:2] )
    print "-" * 80
    print reference, o
    print r_angle
    print o_angle
    print t_angle
    print t_distance
    if t_distance > 10:
      t_distance = 10
    f2 = 0
    min_angle = 1E6
    for i in xrange( len( angles ) ):
      ang = abs( angle.difference( t_angle, angles[i] ) )
      if ang < min_angle:
        f2 = i
        min_angle = ang
    print angles[f2]
    if t_distance < result[f2] or result[f2] == 0:
      result[f2] = t_distance
  return result

def mean( features, weights = None ):  
  if weights == None:
    mu = sum( features * 1.0 / features.shape[0] )
  else:
    p    = weights / np.sum( weights )
    mu = sum( features * np.transpose( np.tile( p, (features.shape[1], 1) ) ) )
  return mu

def difference( v1, v2 ):
  return v2 - v1

def covariance( mean, features, weights = None ):
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  if weights == None:
    return np.dot( np.transpose( diff ) * 1.0 / features.shape[0], diff )
  else:
    p    = weights / np.sum( weights )
    return np.dot( np.transpose( diff ) * np.tile( p, (features.shape[1], 1 ) ), diff )


