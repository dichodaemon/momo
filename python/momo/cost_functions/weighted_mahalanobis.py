import momo
import numpy as np
from math import *

class weighted_mahalanobis( object ):
  def __init__( self, module, p1, p2, dist_sigma = None ):
    self.module = module
    self.inv_dist_sigma = None

    features = momo.compute_features( module, p1, p2 )

    weights = np.ones( (p1.shape[0],) )
    if dist_sigma != None:
      self.inv_dist_sigma = np.linalg.inv( dist_sigma )
      total = 0.0
      for i in xrange( p1.shape[0] ):
        weight = exp( -0.5 * momo.mahalanobis( p1[i, :2], p2[i, :2], self.inv_dist_sigma ) )
        weights[i] = weight
        total += weight
      weights /= total

    self.mu = module.mean( features, weights )
    self.sigma     = module.covariance( self.mu, features, weights )
    self.inv_sigma = np.linalg.inv( self.sigma )

  def __call__( self, v1, v2 ):
    value = self.module.compute( v1, v2 )
    diff  = self.module.difference( self.mu, value )
    cost  = np.dot( np.dot( diff, self.inv_sigma ), np.transpose( diff ) )**0.5
    diff  = v2[:2] - v1[:2]
    decay = 1 
    if self.inv_dist_sigma != None:
      decay = exp( -0.5 * np.dot( np.dot( diff, self.inv_dist_sigma ), np.transpose( diff ) ) ) 
    return cost * decay
