import momo
import numpy as np
import cPickle
from math import *

DEBUG = False

class irl_thrun( object ):
  def __init__( self, module, data = None ):
    self.module = module

    if data != None:
      # Obtain parameters from data
      convert = momo.convert( data, 0.15 )
      frame_data = convert.preprocess_data( data )
      thetas = []
      costs  = []
      for i in xrange( 10 ):
        theta, cost  = momo.irl.learning.thrun.learn( module, convert, frame_data, [46, 47, 48] )
        thetas.append( theta )
        costs.append( cost )
        print "costs", costs
      self.theta = thetas[np.argmax( costs )]

  def __call__( self, s1, s2, frame ):
    value = self.module.compute( s1, s2, frame )
    return np.dot( value, self.w )

  def save( self, stream ):
    pass
    cPickle.dump( [self.module.__name__, self.theta], stream )

  @staticmethod
  def load( stream ):
    module, x, y, width, height, delta, w = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = irl_thrun( module )
    result.theta = theta
    return result

