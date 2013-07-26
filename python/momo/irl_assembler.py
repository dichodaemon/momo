import momo
import numpy as np
import cPickle
from math import *

class irl_assembler( object ):
  def __init__( self, features, learning, radius, h, theta = None, data = None, ids = None, delta = 0.25 ):
    self.features = features
    self.learning = learning
    self.radius = float( radius )
    self.h = int( h )
    self.theta  = theta
    self.delta  = delta

    if data != None:
      self.learn( data, ids )

  def learn( self, data, ids ):
    self.convert = momo.convert( data, self.delta )
    frame_data = self.__convert.preprocess_data( data )

    if ids == None or len( ids ) == 0:
      l = len( frame_data.keys() ) / 2
      ids = range( l, l + 5 )

    self.theta = self.learning.learn( 
      self.features, self.__convert, frame_data, ids, 
      radius = self.radius, h = self.h
    )

  def set_convert( self, convert ):
    self.__convert = convert
    self.compute_costs = self.features.compute_costs( self.__convert, self.radius )
    self.planner = momo.irl.planning.dijkstra( self.__convert, self.compute_costs )

  convert = property( None, set_convert )

  def plan( self, start, goal, velocity, frames ):
    return self.planner( start, goal, velocity, frames[0], self.theta )[0]

  def feature_sum( self, states, frames ):
    result = np.array( [0.] * self.features.FEATURE_LENGTH )
    states = [self.__convert.to_world2( self.__convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
    for i in xrange( len( states ) ):
      result += self.features.compute_feature( states[i], frames[i], self.radius )
    return result

  def save( self, stream ):
    print "Saved", self.features.__name__, self.learning.__name__
    cPickle.dump( [self.features.__name__, self.learning.__name__, self.radius, self.h, self.theta], stream )

  @staticmethod
  def load( stream ):
    features, learning, radius, h, theta = cPickle.load( stream )
    print "Loaded", theta
    features = momo.features.__dict__[features.split( "." )[-1]]
    learning = momo.learning.__dict__[learning.split( "." )[-1]]
    result = irl_assembler( features, learning, radius, h, theta )
    return result

