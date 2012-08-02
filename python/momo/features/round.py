from momo import *
from default import *

import numpy as np
from math import *

feature_size = 15
angles    = [0, pi / 4, 3 * pi / 4, -pi / 4, -3 * pi / 4]
#feature_size = 24
#angles = [0, pi]
#for i in xrange( 1, 4 ):
  #angles.append( pi * i * 0.25 )
  #angles.append( - pi * i * 0.25 )
n = len( angles )

def compute( reference, frame ):
  result = np.array( [0.] * n * 3 )
  r_angle = angle.as_angle( reference[2:] )
  #print "-" * 80
  #print reference
  #print frame
  for o in frame:
    o_angle = angle.as_angle( o[:2] - reference[:2] )
    t_angle = angle.difference( r_angle, o_angle )
    v_angle = angle.rotate( o[2:], -r_angle )
    t_distance = distance( o[:2], reference[:2] )
    #print reference, o
    #print r_angle
    #print o_angle
    #print t_angle
    #print t_distance
    if t_distance > 10:
      t_distance = 10
    f2 = 0
    min_angle = 1E6
    for i in xrange( len( angles ) ):
      ang = abs( angle.difference( t_angle, angles[i] ) )
      if ang < min_angle:
        f2 = i
        min_angle = ang
    #print angles[f2]
    if t_distance < result[f2] or result[f2] <= 0:
      #print "o", f2, t_distance, o
      result[f2] = t_distance
      result[n + f2] = v_angle[0]
      result[n * 2 + f2] = v_angle[1]
  #print result
  for i in xrange( n ):
    if result[i] <= 0:
      result[i] = 5.
  return result

