import momo
import numpy as np
from math import *
import random

class gaussian_mixture( object ):
  def __init__( self, module, p1, p2, k ):
    self.k      = k
    self.module = module

    features = momo.compute_features( module, p1, p2 )
    
    self.mu    = random.sample( features, k )
    self.sigma = []
    self.inv_sigma = []
    self.prior = np.array( [ 1.0 / k ] * k )
    for i in xrange( k ):
      self.sigma.append( np.eye( module.feature_size ) )
      self.inv_sigma.append( np.eye( module.feature_size ) )

    for i in xrange( 20 ):
      self.optimize( features )

  def optimize( self, features ):
    # Expectation step
    expectations = [0.0] * len( features )
    for fi in xrange( len( features ) ):
      expectations[fi] = [0.0] * self.k 
      z = 0.0
      for ki in xrange( self.k ):
        diff = self.module.difference( self.mu[ki], features[fi] )
        maha = np.dot( np.dot( diff, self.inv_sigma[ki] ), np.transpose( diff ) )
        tmp  = exp( -0.5 * maha ) * self.prior[ki]
        expectations[fi][ki] = tmp
        z += tmp
      for ki in xrange( self.k ):
        expectations[fi][ki] /= z

    # Maximization step
    for ki in xrange( self.k ):
      weights = np.ones( ( len( features ),) )
      z = 0.0
      for fi in xrange( len( features ) ):
        weights[fi] = expectations[fi][ki]
        z += weights[fi]
      weights *= 1.0 / z

      self.mu[ki] = self.module.mean( features, weights )
      self.sigma[ki] = self.module.covariance( self.mu[ki], features, weights )
      self.inv_sigma[ki] = np.linalg.inv( self.sigma[ki] )
      self.prior[ki] = z / len( features )

  def __call__( self, v1, v2 ):
    value = self.module.compute( v1, v2 )
    cost  = 0.0
    for ki in xrange( self.k ):
      diff  = self.module.difference( self.mu[ki], value )
      cost += (self.prior[ki] * np.dot( np.dot( diff, self.inv_sigma[ki] ), np.transpose( diff ) ) )**0.5 
    return cost





