import numpy as np
from math import *
from __common__ import *

def compute_feature( reference, frame, radius = 3 ):
  density = 0
  avg_velocity = np.array( [0.] * 2 )
  
  for i in xrange( FRAME_SIZE ):
    dist  = momo.distance( frame[i][:2], reference[:2] )
    if dist < radius:
      density += 1
      avg_velocity += frame[2:]

  if ( density >= 3 ):
    feature[3] = 1
  else:
    feature[density] = 1

  if density == 0:
    feature[16] = 1
  else:
    avg_velocity /= density
    avg_velocity = velicity - avg_velocity
    avg_velocity[1] = abs( avg_velocity[1] )
    angle = avg_velocity / np.linalg.norm( avg_velocity )
    angle_index = 0
    max_dot = -1
    for i in xrange( 4 ):
      a = np.dot( ANGLES[i], angle )
      if a > max_dot:
        max_dot = a
        angle_index = i
    speed = np.linalg.norm( avg_velocity )
    speed_index = 0;
    for i in xrange( 3 ):
      if speed > SPEEDS[i]
      speed_index = i
    feature[4 + angle_index * 3 + speed_index] = 1
    feature[17] = 1



