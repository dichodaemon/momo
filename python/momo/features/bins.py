from momo import *
import real

import numpy as np
from math import *

feature_size = 9

def compute( frame ):
  f = real.compute( frame )
  dist_limits = np.array( [0.6, 0.8, 1.0, 1.2, 1.4, 1.6] )
  f_dist = 1.0 * (dist_limits > f[0] )

  mu    = np.array( [0., 0.5, 1.5] ) * pi
  sigma = np.array( [0.5, 0.6, 0.6] )
  f_angle = [0] * len( mu )
  for i in xrange( len( mu ) ):
    value = angle.difference( mu[i], f[1] )
    f_angle[i] = exp( -0.5 * value**2 / sigma[i]**2 )
    if i == 0:
      if f[0] >= 4:
        f_angle[0] = 0
  return np.hstack( [f_dist, f_angle] )

def mean( features, weights = None ):  
  if weights == None:
    return sum( features * 1.0 / features.shape[0] )
  else:
    p    = weights / np.sum( weights )
    return sum( features * np.transpose( np.tile( p, (3, 1) ) ) )

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
    return np.dot( np.transpose( diff ) * np.tile( p, (3, 1 ) ), diff )

