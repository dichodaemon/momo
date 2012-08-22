import momo
import numpy as np
from math import *
import random

class irl_j1( object ):
  def __init__( self, module, data = None ):
    self.module = module
    self.values = np.array( [
      7.4296385088625927e-01, 2.6371672429264259e-01, 1.9394104274488257e+00, 0.0000000000000000e+00,  
      1.0004851091543651e+01, 1.0345848529954143e+01, 0.0000000000000000e+00, 
      0.0000000000000000e+00, 1.2242664576625843e-01, 0.0000000000000000e+00, 
      4.4204046143653342e+00, 3.4393845914652230e+00, 0.0000000000000000e+00, 
      0.0000000000000000e+00, 1.5307974231832036e+00, 2.4769656817494350e+00, 
      0.0000000000000000e+00
    ] )

  def __call__( self, s1, s2, frame ):
    value = self.module.compute( s1, s2, frame )
    return np.dot( value, self.values )

  def save( self, stream ):
    pass


  @staticmethod
  def load( stream ):
    return irl_j1( momo.features.flow2 )


