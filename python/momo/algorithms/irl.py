import momo
import numpy as np
import cPickle
from math import *

class irl( object ):
  def __init__( self, learning, radius, replan, theta = None, data = None, ids = None, delta = 0.25 ):
    self.features = momo.irl.features.flow
    self.learning = learning
    self.__learn = momo.irl.learning.__dict__[learning].learn
    self.radius = float( radius )
    self.replan = int( replan )
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

    self.theta = self.__learn( 
      self.features, self.__convert, self.planner, frame_data, ids, 
      radius = self.radius, replan = self.replan
    )

  def set_convert( self, convert ):
    self.__convert = convert
    self.compute_costs = self.features.compute_costs( self.__convert, self.radius )
    self.planner = momo.irl.planning.dijkstra( self.__convert, self.compute_costs )

  convert = property( None, set_convert )

  def plan( self, start, goal, velocity, frames, replan = 1 ):
    return self.planner( start, goal, velocity, frames, replan, self.theta )

  def feature_sum( self, states, frames ):
    result = np.array( [0.] * self.features.FEATURE_LENGTH )
    states = [self.__convert.to_world2( self.__convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
    for i in xrange( len( states ) ):
      result += self.features.compute_feature( states[i], frames[i], self.radius )
    return result

  def save( self, stream ):
    cPickle.dump( [self.learning, self.radius, self.replan, self.theta], stream )

  @staticmethod
  def load( stream ):
    learning, radius, replan, theta = cPickle.load( stream )
    result = irl( learning, radius, replan, theta )
    return result
