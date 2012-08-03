from numbers import *
from math import *
import numpy as np

def as_vector( value ):
  if isinstance( value, Number ):
    return np.array( [cos( value ), sin( value )] )
  else:
    norm = np.dot( value, value )**0.5
    if norm != 0:
      return value / norm
    else:
      return np.array( [1., 0] )

def as_angle( vector ):
  return atan2( vector[1], vector[0] )

def difference( angle1, angle2 ):
  if isinstance( angle1, Number ):
    angle1 = as_vector( angle1 )
    angle2 = as_vector( angle2 )
  angle = atan2( angle2[1], angle2[0] ) - atan2( angle1[1], angle1[0] )
  while abs( angle ) > 2 * pi:
    angle = np.sign( angle ) * ( abs( angle ) - 2 * pi )
  if abs( angle ) > pi:
    angle = -np.sign( angle ) * ( 2 * pi - abs( angle ) )
  return angle

def rotate( v, angle ):
  x = v[0] * cos( angle ) - v[1] * sin( angle )
  y = v[0] * sin( angle ) + v[1] * cos( angle )
  return np.array( [x, y] )

if __name__ == "__main__":
  angle1 = np.array( [1., 0] )
  for i in xrange( 360 ):
    angle  =  i * 2 * pi / 300
    angle2 = np.array( [cos( angle ), sin( angle )] )
    print angle, angle2, difference( angle1, angle2 )

