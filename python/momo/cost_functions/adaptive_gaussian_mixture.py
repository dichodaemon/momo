import momo
import numpy as np
from math import *
import random

class adaptive_gaussian_mixture( object ):
  def __init__( self, module, data, kmax = None ):
    self.module = module
    features = momo.compute_features( module, data )

    print len( features )
    self.agmm = momo.clustering.adaptive_gaussian_mixture_model( module )
    self.agmm.learn( features, kmax = kmax )
    
    self.mu    = self.agmm.mu
    self.sigma = self.agmm.sigma
    self.inv_sigma = self.agmm.inv_sigma
    self.prior = self.agmm.prior
    self.k = self.agmm.k
    self.norm = []
    dim  = self.mu[0].shape[0]
    for i in xrange( self.k ):
      det   = np.linalg.det( self.sigma[i] )
      norm  = ( 2 * pi )**( -dim * 0.5 ) * det**( -0.5 )
      self.norm.append( norm )
    print "%i clusters" % self.k

  def __call__( self, frame ):
    value = self.module.compute( frame )
    z     = 0.0
    result = 0

    for ki in xrange( self.k ):
      diff  = self.module.difference( self.mu[ki], value )
      maha  = np.dot( np.dot( diff, self.inv_sigma[ki] ), np.transpose( diff ) )
      proba = self.prior[ki] * self.norm[ki] * exp( -0.5 * maha )
      z += proba
      #result += proba * maha**0.5
      #result += proba
      result +=  self.prior[ki] * maha**0.5
    #return (-log( result ) )**0.5
    #return result / z
    return result
    #return (-log( result ) )**0.5

