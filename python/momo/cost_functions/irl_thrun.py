import momo
import numpy as np
import cPickle
from math import *

DEBUG = False

class irl_thrun( object ):
  def __init__( self, module, data = None ):
    self.module = module

    if data != None:
      convert = momo.convert( data, 0.25 )
      frame_data = convert.preprocess_data( data )
      m = len( frame_data.keys() ) / 2
      #self.theta = momo.irl.learning.thrun.learn( module, convert, frame_data, range( m, m + 5 ), radius = 3 )
      self.theta = momo.irl.learning.thrun.learn( module, convert, frame_data, [62, 168], radius = 3 )

  def __call__( self, s1, s2, frame ):
    value = self.module.compute( s1, s2, frame )
    return np.dot( value, self.w )

  def save( self, stream ):
    pass
    cPickle.dump( [self.module.__name__, self.theta], stream )

  @staticmethod
  def load( stream ):
    module, theta = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = irl_thrun( module )
    result.theta = theta
    return result

