import momo
import numpy as np
from math import *
import random

class adaptive_gaussian_mixture( object ):
  def __init__( self, module, p1, p2 ):
    self.module = module
    features = momo.compute_features( module, p1, p2 )

    self.agmm = momo.clustering.adaptive_gaussian_mixture_model( module )
    self.agmm.learn( features )
    
    self.mu    = self.agmm.mu
    self.sigma = self.agmm.sigma
    self.inv_sigma = self.agmm.inv_sigma
    self.prior = self.agmm.prior
    self.k = self.agmm.k
    print "%i clusters" % self.k

  def __call__( self, v1, v2 ):
    value = self.module.compute( v1, v2 )
    cost  = 0.0
    for ki in xrange( self.k ):
      diff  = self.module.difference( self.mu[ki], value )
      cost += (self.prior[ki] * np.dot( np.dot( diff, self.inv_sigma[ki] ), np.transpose( diff ) ) )**0.5 
    return cost

