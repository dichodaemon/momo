import numpy as np

class Kalman( object ):
  def __init__( self, mean, system_matrix, observation_matrix, system_noise, observation_noise, state_covariance ):
    self.__mean = mean
    self.a = system_matrix
    self.c = observation_matrix
    self.r = system_noise
    self.q = observation_noise
    self.p = state_covariance

  def predict( self, delta ):
    self.__mean = np.dot( self.a, self.__mean )
    self.p = np.dot( np.dot( self.a, self.p ), np.transpose( self.a ) ) + self.r( delta )

  def update( self, observation ):
    tmp = np.dot( np.dot( self.c, self.p ), np.transpose( self.c ) ) + self.q
    tmp = np.linalg.inv( tmp )
    k = np.dot( np.dot( self.p, np.transpose( self.c ) ), tmp )
    self.__mean = self.__mean + np.dot( k, observation - np.dot( self.c, self.__mean ) )
    self.p = np.dot( np.eye( 4 ) - np.dot( k, self.c ), self.p )

  def get_mean( self ):
    return self.__mean

  def set_mean( self, value ):
    self.__mean = value

  mean = property( get_mean, set_mean )

