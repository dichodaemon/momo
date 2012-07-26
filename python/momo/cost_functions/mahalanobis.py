import momo
import numpy as np
import cPickle
from math import *

class mahalanobis( object ):
  def __init__( self, module, data = None ):
    self.module = module
    if data != None:
      features = momo.compute_features( module, data )

      self.mu = module.mean( features )
      self.sigma     = module.covariance( self.mu, features )
      self.inv_sigma = np.linalg.inv( self.sigma )

  def __call__( self, reference, frame ):
    value = self.module.compute( reference, frame )
    diff  = self.module.difference( self.mu, value )
    cost  = np.dot( np.dot( diff, self.inv_sigma ), np.transpose( diff ) )**0.5
    return 1.0
    return cost

  def save( self, stream ):
    cPickle.dump( [self.module.__name__, self.mu, self.sigma, self.inv_sigma], stream )

  @staticmethod
  def load( stream ):
    module, mu, sigma, inv_sigma = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = mahalanobis( module )
    result.mu = mu
    result.sigma = sigma
    result.inv_sigma = inv_sigma
    return result

