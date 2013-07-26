import numpy as np
import momo
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH

  # Initialize weight vector
  w  = ( np.ones( feature_length ) * 1.0 / feature_length ).astype( np.float64 )

  total_t = 0
  for o_id in ids:
    total_t += len( frame_data[o_id]["states"] ) - h
  gamma = 2.0
  inc   = gamma / total_t

  for o_id in ids:
    l = len( frame_data[o_id]["states"] )
    for i in xrange( max( l - h, 1 ) ):
      observed, expected, cummulated, costs, grid_path =\
        momo.learning.max_ent.compute_expectations( 
          feature_module, convert, radius,
          frame_data[o_id]["states"][i:], frame_data[o_id]["frames"][i:],
          w, h
        )
      gradient = observed / np.sum( observed[:4] ) - expected / np.sum( expected[:4] )
      error = np.linalg.norm( gradient )
      #momo.plot.gradient_descent_step( cummulated, costs, grid_path, error )
      print o_id, gamma, error
      if np.any( np.isnan( gradient ) ):
        continue
      for i in xrange( feature_length ):
        w[i] *= exp( -gamma * gradient[i] )
      gamma -= inc

  return w

