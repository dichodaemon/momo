import momo
import numpy as np
from math import *

class mahalanobis( object ):
  def __init__( self, module, data ):
    self.module = module
    self.inv_dist_sigma = None

    features = momo.compute_features( module, data )

    self.mu = module.mean( features )
    self.sigma     = module.covariance( self.mu, features )
    self.inv_sigma = np.linalg.inv( self.sigma )

  def __call__( self, v1, v2 ):
    value = self.module.compute( v1, v2 )
    diff  = self.module.difference( self.mu, value )
    cost  = np.dot( np.dot( diff, self.inv_sigma ), np.transpose( diff ) )**0.5
    diff  = v2[:2] - v1[:2]
    decay = 1 
    return cost * decay

