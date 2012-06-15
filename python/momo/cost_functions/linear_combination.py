import momo
import numpy as np
from math import *

class linear_combination( object ):
  def __init__( self, values ):
    self.values = values

  def __call__( self, v1, v2 ):
    value = momo.features.bins.compute( v1, v2 )
    return np.dot( value, self.values )


