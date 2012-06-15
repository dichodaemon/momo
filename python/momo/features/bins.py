from momo import *
import real

import numpy as np
from math import *

feature_size = 9

def compute( p1, p2 ):
  f = real.compute( p1, p2 )
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

def mean( features, weights = 1.0 ):  
  p    = weights / np.sum( weights )
  mu = sum( features * np.transpose( np.tile( p, (feature_size, 1) ) ) )
  return mu

def difference( v1, v2 ):
  return v2 - v1

def covariance( mean, features, weights ):
  p = weights / np.sum( weights )
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  return np.dot( np.dot( np.transpose( diff ), np.diag( p ) ), diff )



